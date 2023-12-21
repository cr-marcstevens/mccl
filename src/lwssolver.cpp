#include <mccl/config/config.hpp>
#include <mccl/config/utils.hpp>

#include <mccl/algorithm/decoding.hpp>
#include <mccl/algorithm/isdgeneric.hpp>
#include <mccl/algorithm/lowweight_generic.hpp>
#include <mccl/algorithm/wagner.hpp>

#include <mccl/tools/parser.hpp>
#include <mccl/tools/generator.hpp>
#include <mccl/tools/statistics.hpp>
#include <mccl/tools/utils.hpp>

#include <mccl/contrib/program_options.hpp>
#include <mccl/contrib/string_algo.hpp>

#include <iostream>
#include <cstdlib>
#include <memory>

using namespace mccl;

/* run Trials */

void run_LWS(lowweight_search_API& LWS, cmat_view& G, size_t w, bool quiet)
{
  LWS.initialize(G, w);
  LWS.solve();
  if (!quiet)
    std::cout << "Solution found:\n" << LWS.get_solution() << std::endl;
}

void runtrials_LWS(lowweight_search_API& LWS, cmat_view& G, size_t w, size_t trials, bool quiet, bool generate, SDP_generator& generator)
{
  // run trials
  time_statistic time_trial_stat, time_total_stat;

  time_total_stat.start();
  for (size_t i = 0; i < trials; ++i)
  {
    if(i > 0 && generate)
    {
      generator.generate(G.columns(), G.rows(), w);
      G.reset(generator.G());
    }
    time_trial_stat.start();
    run_LWS(LWS, G,w, quiet);
    time_trial_stat.stop();
  }
  time_total_stat.stop();

  /* print basic overall statistics */
  double total_time = time_total_stat.total(), avg_time = time_trial_stat.mean();
  double avg_loop_cnt = LWS.get_stats().cnt_loop_next.mean(),
         total_loop_cnt = LWS.get_stats().cnt_loop_next.total();

  std::cout << "=== Basic statistics ===" << std::endl;
  std::cout << "  Time                 : mean= " << std::setw(10) << avg_time     << "s  total= " << std::setw(10) << total_time << "s" << std::endl;
  std::cout << "  Number of iterations : mean= " << std::setw(10) << avg_loop_cnt << "   total= " << std::setw(10) << total_loop_cnt << std::endl;
  std::cout << "  Inverse of iterations: mean= " << std::setw(10) << 1.0/avg_loop_cnt << std::endl;
  std::cout << "  Time per iteration   : mean= " << std::setw(10) << avg_time/avg_loop_cnt << "s" << std::endl;
}



/* run Benchmark */

void benchmark_LWS(lowweight_search_API& LWS, cmat_view& G, size_t w, size_t min_iterations, double min_total_time)
{
  LWS.initialize(G, w);
  LWS.prepare_loop(true);
  
  size_t its = min_iterations, total_its = 0;
  time_statistic total_time;
  total_time.start();
  while (true)
  {
    for (size_t i = 0; i < its; ++i)
      LWS.loop_next();
      
    total_its += its;
    double total_elapsed_time = total_time.elapsed_time();
    
    if (total_elapsed_time >= min_total_time)
      break;
    // if measured time is very small, try at least 1000 x the number of iterations
    if (total_elapsed_time <= 0.0)
    {
      its *= 1000;
      continue;
    }
    its = size_t( double(total_its) * (min_total_time * 1.25) / total_elapsed_time ) - total_its;
  }
  total_time.stop();
  
  std::cout << "Time                 : " << total_time.total() << "s" << std::endl;
  std::cout << "Number of iterations : " << total_its << std::endl;
  std::cout << "Time per iteration   : mean=" << total_time.total()/double(total_its) << "s" << std::endl;
}


/* Main program */

int main(int argc, char *argv[])
{
//try
//{
    /* Configuration variables */
    std::string filepath, algo;
    size_t trials;
    bool quiet = false;
    bool print_stats = false;
    bool print_input = false;
    
    // generator options
    int n = 0, k, w;
    uint64_t genseed;
    
    // benchmark options
    bool benchmark = false;
    size_t min_bench_iterations;
    double min_bench_time;
    
    // maximum width to print program options
    unsigned line_length = 78;
    
    
    /* Define program options */
    po::options_description
      allopts,
      cmdopts("Command options", line_length, line_length/2),
      auxopts("Extra options", line_length, line_length/2),
      genopts("Generator options", line_length, line_length/2),
      benchopts("Benchmark options", line_length, line_length/2),
      lwsopts("LWS options")
      ;
      
    // These are the core commands, you need at least one of these
    cmdopts.add_options()
      ("help,h", "Show options")
      ("manual", "Show manual")
      ("file,f", po::value<std::string>(&filepath), "Specify input file")
      ("generate,g", "Generate random LWS instances")
      ;
    // these are other configuration options
    auxopts.add_options()
      ("algo,a", po::value<std::string>(&algo)->default_value("W"), "Specify algorithm: W")
      ("trials,t", po::value<size_t>(&trials)->default_value(1), "Number of LWS trials")
      ("quiet,q", po::bool_switch(&quiet), "Quiet: reduce verbosity of trials")
      ("printinput", po::bool_switch(&print_input), "Print input G")
      ("printstats", po::bool_switch(&print_stats), "Print LWS function call statistics")
      ;
    // options for the generator
    genopts.add_options()
      ("genunique", "Generate unique decoding instance")
      ("genrandom", "Generate random decoding instance")
      ("genseed", po::value<uint64_t>(&genseed), "Set instance generator random generator seed")
      ("n", po::value<int>(&n), "Code length")
      ("k", po::value<int>(&k)->default_value(-1), "Code dimension ( -1 = auto with rate 1/2 )")
      ("w", po::value<int>(&w)->default_value(-1), "Error weight ( -1 = 1.05*d_GV )")
      ;
    benchopts.add_options()
      ("benchmark", po::bool_switch(&benchmark), "Instead of many trials, benchmark LWS iterations on 1 instance")
      ("minbenchits", po::value<size_t>(&min_bench_iterations)->default_value(100), "Minimum number of LWS iterations")
      ("minbenchtime", po::value<double>(&min_bench_time)->default_value(100.0), "Minimal total time (s) for benchmark")
      ;

    /* Collect submodule program options */
    std::vector< std::unique_ptr<module_configuration_API> > modules;
    
    // ========== ADD MODULE DEFAULT CONFIGURATIONS HERE ===============
    modules.emplace_back( make_module_configuration( lowweight_generic_config_default ) );
    modules.emplace_back( make_module_configuration( wagner_config_default ) );

    // =================================================================
    
    //  if there are common options then only the first description is used
    //  any default values are ignored, so if no value is passed each algorithm can use its own default value
    for (auto& ptr : modules)
      ptr->options_description_insert(lwsopts);

    /* Parse all program options */
    allopts.add(cmdopts).add(auxopts).add(genopts).add(benchopts).add(lwsopts);
    po::variables_map vm;
    // TODO: configuration file?
    // parse command line parameters
    po::store(po::parse_command_line(argc, argv, allopts, false, true /*allow positional parameters*/), vm);
    po::notify(vm);

    // process positional parameters if any
    if (vm.positional.size() > 0)
      n = vm.positional[0].as<size_t>();
    if (vm.positional.size() > 1)
      k = vm.positional[1].as<int>();
    if (vm.positional.size() > 2)
      w = vm.positional[2].as<int>();
    if (vm.positional.size() > 3)
    {
      std::cout << "Unknown option: " << vm.positional[3].as<std::string>() << std::endl;
      return 1;
    }

    // store configuration in configmap
    configmap_t configmap;
    for (auto& o : vm)
    {
      if (o.second.empty())
        configmap[o.first];
      else
        configmap[o.first] = o.second.as<std::string>();
    }

    // pass configmap to submodules
    for (auto& ptr : modules)
      ptr->load_config(configmap);

    /* show help and/or manual if requested or if no command was given */
    if (vm.count("help") || vm.count("manual") ||
        vm.count("file")+vm.count("generate")==0
        )
    {
      std::vector<po::options_description> vec_opts({ cmdopts, auxopts, genopts, benchopts });
      for (auto& ptr : modules)
        vec_opts.emplace_back(ptr->get_options_description(line_length));
      po::print_options_description(vec_opts.begin(), vec_opts.end());

      if (vm.count("manual"))
      {
        std::cout << "\n\n === LWS solver manual ===\n";
        
        for (auto& ptr: modules)
          ptr->print_manual();
      }

      return 0;
    }


    /* Create the corresponding syndrome decoding object */
    std::unique_ptr<lowweight_search_API> LWS_ptr;
    std::unique_ptr<subISDT_API> subISD_ptr;
    std::string LWS_conf_str, subISD_conf_str;
    

#define INITIALIZE_ALGO(subISDT_type) \
    auto _subISD = new subISDT_type(); \
    auto _LWS = new lowweight_generic<subISDT_type>(*_subISD); \
    subISD_ptr.reset(_subISD); \
    LWS_ptr.reset(_LWS); \
    subISD_conf_str = get_configuration_str(*_subISD); \
    LWS_conf_str = get_configuration_str(*_LWS);


    // ==================== ADD NEW ALGORITHMS HERE ====================
    // NOTE: algorithms default configuration needs to be added above in 'modules'
    // initialize chosen algorithm and set pretty print name of algorithm
    sa::to_upper(algo); // make input algo entirely upper-case
    if (algo == "W" || algo == "Wagner")
    {
      algo = "Wagner";
      INITIALIZE_ALGO( subISDT_wagner );
    }
    else
    {
      std::cout << "Unknown algorithm: " << algo << std::endl;
      return 1;
    }
    // =================================================================

    /* parse or generate instances */
    cmat_view G;

    file_parser parser;
    SDP_generator generator;
    if (vm.count("genseed"))
      generator.seed(genseed);
    genseed = generator.get_seed();

    if (!filepath.empty())
    {
      std::cout << "Loading file: " << filepath << "..." << std::flush;
      parser.parse_file(filepath);
      std::cout << "done." << std::endl;
      n = parser.n();
      k = parser.k();
      // allow for manual override of w and for missing w
      if (w < 0)
        w = parser.w();
      if (w < 0)
        w = get_cryptographic_w(n, k);
      G.reset(parser.G());
    }
    else
    {
      if (k < 0)
        k = n>>1;
      if (w < 0)
        w = get_cryptographic_w(n, k);
      if (n <= 0 || k >= n || w >= n)
      {
        std::cout << "Bad input parameters: n=" << n << ", k=" << k << ", w=" << w << std::endl;
        return 1;
      }
      generator.generate(n, k, w);
      G.reset(generator.G());
    }
    
    std::cout << "Run settings       : n=" << n << " k=" << k << " w=" << w << " trials=" << trials;
    if (vm.count("generate"))
      std::cout << " genseed=" << genseed;
    std::cout << std::endl;
    std::cout << " -     LWS generic : " << LWS_conf_str << std::endl;
    std::cout << " - " << std::setw(15) << algo << " : " << subISD_conf_str << std::endl;
    
    if (print_input)
    {
      std::cout << "G = \n" << G << std::endl;
    }

    /* run all trials / benchmark */
    if (benchmark)
    {
      // run benchmark
      if (min_bench_iterations == 0)
        min_bench_iterations = 1;
      if (min_bench_time <= 1.0)
        min_bench_time = 1.0;
      benchmark_LWS(*LWS_ptr, G,w, min_bench_iterations, min_bench_time);
    }
    else
    {
      runtrials_LWS(*LWS_ptr, G,w, trials, quiet, vm.count("generate"), generator);
    }

    /* print detailed statistics */
    if (print_stats)
    {
      std::cout << "\n=== Detailed statistics ===" << std::endl;
      LWS_ptr->get_stats().print(std::cout);
      subISD_ptr->get_stats().print(std::cout);
    }
    
    return 0;
/*
}
catch (std::exception& e)
{
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return 1;
}
catch (...)
{
    std::cerr << "Caught unknown exception!" << std::endl;
    return 1;
}*/
}