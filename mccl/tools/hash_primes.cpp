#include <mccl/tools/hash_primes.hpp>

MCCL_BEGIN_NAMESPACE

namespace detail {

struct hash_prime_t
{
    uint64_t prime;
    uint64_t muldiv;
    unsigned shift;
};

const hash_prime_t hash_prime_table[] = {
    { 3, 0xaaaaaaaaaaaaaaab, 1 },
    { 5, 0xcccccccccccccccd, 2 },
    { 11, 0x2e8ba2e8ba2e8ba3, 1 },
    { 13, 0x4ec4ec4ec4ec4ec5, 2 },
    { 17, 0xf0f0f0f0f0f0f0f1, 4 },
    { 19, 0xd79435e50d79435f, 4 },
    { 37, 0xdd67c8a60dd67c8b, 5 },
    { 41, 0xc7ce0c7ce0c7ce0d, 5 },
    { 59, 0x8ad8f2fba9386823, 5 },
    { 67, 0xf4898d5f85bb3951, 6 },
    { 73, 0x70381c0e070381c1, 5 },
    { 83, 0x3159721ed7e75347, 4 },
    { 109, 0x964fda6c0964fda7, 6 },
    { 113, 0x90fdbc090fdbc091, 6 },
    { 131, 0x3e88cb3c9484e2b, 1 },
    { 149, 0x1b7d6c3dda338b2b, 4 },
    { 163, 0xc907da4e871146ad, 7 },
    { 179, 0xb70fbb5a19be3659, 7 },
    { 197, 0x14cab88725af6e75, 4 },
    { 227, 0x905a38633e06c43b, 7 },
    { 241, 0x10fef010fef010ff, 4 },
    { 257, 0xff00ff00ff00ff01, 8 },
    { 283, 0x73c9b97112ff186d, 7 },
    { 311, 0x34ae820ed114942b, 6 },
    { 349, 0xbbc8408cd63069a1, 8 },
    { 383, 0x558e5ee9f14b87b, 3 },
    { 421, 0x9baade8e4a2f6e1, 4 },
    { 499, 0x8355ace3c897db1, 4 },
    { 509, 0x10182436517a3753, 5 },
    { 521, 0xfb93e672fa98528d, 9 },
    { 557, 0xeb51599f7ba23d97, 9 },
    { 613, 0xd5d20fde972d8539, 9 },
    { 677, 0xc19b6a41cbd11c5d, 9 },
    { 751, 0xae87ab7648f2b4ab, 9 },
    { 827, 0x9e7dada8b4c75a15, 9 },
    { 941, 0x22d291467611f493, 7 },
    { 1013, 0x8163d282e7fdfa71, 9 },
    { 1031, 0x3f90c2ab542cb1c9, 8 },
    { 1039, 0xfc4ddc06e6210431, 10 },
    { 1151, 0x71e06ac264163dd5, 9 },
    { 1277, 0xcd47f7fb3050301d, 10 },
    { 1409, 0x5d065bef48db7b01, 9 },
    { 1549, 0x2a4eff8113017cc7, 8 },
    { 1709, 0x4cb1f4ea479a23a7, 9 },
    { 1879, 0x22e0cce8b3d7209, 4 },
    { 2029, 0x204cb630b3aab56f, 8 },
    { 2053, 0xff6063c1a6f7a539, 11 },
    { 2069, 0xfd66d2187fb0cfdf, 11 },
    { 2281, 0x3976677a38571775, 9 },
    { 2521, 0x33fdf8144f34e7ef, 9 },
    { 2789, 0x5dfdfb0b1b42ea1, 6 },
    { 3067, 0xaaf1e4c9fed4d8b, 7 },
    { 3373, 0x26dbf2f21c62aa77, 9 },
    { 3727, 0x4656227b39e768e3, 10 },
    { 4091, 0x80280c83e938e1c7, 11 },
    { 4099, 0xffd008fe5050f0d3, 12 },
    { 4513, 0xe8587db3e001d0b1, 12 },
    { 4967, 0x698de3dbec009e55, 11 },
    { 5471, 0x2fea49d68ac91cdf, 10 },
    { 6037, 0xadb10aa4c956f917, 12 },
    { 6659, 0x13aef5a893eeee47, 9 },
    { 7331, 0x8f087c50e00c4abb, 12 },
    { 8081, 0x20708651ec2b35e3, 10 },
    { 8179, 0x80341528987df32b, 12 },
    { 8209, 0x7fbc240cd92ca04b, 12 },
    { 8893, 0x75e90739b7a15971, 12 },
    { 9791, 0xd6311a61bc47d9b9, 13 },
    { 10771, 0x185683878bd30827, 10 },
    { 11887, 0x2c1b22b1d86aa59d, 11 },
    { 13093, 0x5016362905607dc3, 12 },
    { 14411, 0x48c31f3f4b3b3e5f, 12 },
    { 15859, 0x421e61356a2ae7f7, 12 },
    { 16381, 0x4003002401b01441, 12 },
    { 16411, 0x1ff285af99eb10d5, 11 },
    { 17467, 0xf020986cb0c0fe33, 14 },
    { 19219, 0xda3cc43b83b2437b, 14 },
    { 21143, 0xc660be3dc6703dcd, 14 },
    { 23269, 0xb440bbff84137ec1, 14 },
    { 25601, 0xa3d566d373a53e59, 14 },
    { 28163, 0x94edf9828118681, 10 },
    { 30983, 0x875fd67d1cbaa2b1, 14 },
    { 32749, 0x801302d26b3beae5, 14 },
    { 32771, 0xfffa0023ff28051, 11 },
    { 34123, 0x3d75672dc1a04939, 13 },
    { 37537, 0xdf79c89bc472c413, 15 },
    { 41299, 0x32c79c467dd8905b, 13 },
    { 45491, 0xb866c7c97b1cce9f, 15 },
    { 50047, 0x53ceab498d24bb71, 14 },
    { 55051, 0x9860fc3a8981e51d, 15 },
    { 60607, 0x8a68ee54cee3687f, 15 },
    { 65449, 0x4015c766c3ec9567, 14 },
    { 65537, 0xffff0000ffff0001, 16 },
    { 66697, 0x3ee2cd6a686c6c49, 14 },
    { 73369, 0xe4ab43b549fb54d9, 16 },
    { 80737, 0x33f340e0a4e18b69, 14 },
    { 88811, 0xbce8c21906adc6a5, 16 },
    { 97711, 0xabb3d25c2fb1a703, 16 },
    { 107509, 0x9c0dd6ea333d1347, 16 },
    { 118259, 0x8dde4ff3d7c3060b, 16 },
    { 130099, 0x80f511ba3054d93f, 16 },
    { 131059, 0x800340152089537d, 16 },
    { 131101, 0x3ffc60348d060329, 15 },
    { 143111, 0x3a9db86a3a346503, 15 },
    { 157427, 0x6a92475bd63be421, 16 },
    { 173177, 0x30708357121e7601, 15 },
    { 190523, 0xb01e13a2ea7a7b1b, 17 },
    { 209579, 0xa01a9e6cf6fdd093, 17 },
    { 230561, 0x9188aaf708b70ba1, 17 },
    { 253637, 0x844b0a68b9832a6d, 17 },
    { 262133, 0x80016003c80a661d, 17 },
    { 262147, 0xffff40008fff9401, 18 },
    { 279001, 0xf0885f110602cc6f, 18 },
    { 306913, 0x6d542caa4177565b, 17 },
    { 337607, 0xc6c72ed7b6a421e1, 18 },
    { 371383, 0x1696656d5f7b5d5d, 15 },
    { 408539, 0xa443f7f39f78f33f, 18 },
    { 449399, 0x4aaa458ec1ceaee3, 17 },
    { 494369, 0x87bf1af5fe7291ff, 18 },
    { 524269, 0x80013002d206b2d, 14 },
    { 524309, 0xfffd6006e3ede9b, 15 },
    { 543811, 0x7b679e1e15f37ef3, 18 },
    { 598193, 0x702f9bf44af820b5, 18 },
    { 658043, 0xcbf708fedf4830a5, 19 },
    { 723851, 0xb96bf89bc1a56e7f, 19 },
    { 796267, 0xa88f06c4952430e3, 19 },
    { 875893, 0x993c3cb94d66446b, 19 },
    { 963497, 0x1169afabd90e55b, 12 },
    { 1048559, 0x8000880090809989, 19 },
    { 1048583, 0xffff900030ffea91, 20 },
    { 1059847, 0x3f51c372bef0b681, 18 },
    { 1165831, 0x7320509e2cf40373, 19 },
    { 1282417, 0x68a8f3f5cb62720d, 19 },
    { 1410679, 0x17c938492b1d8033, 17 },
    { 1551757, 0x567e793c3d67c8d5, 19 },
    { 1706951, 0x4ea14e3d85495af7, 19 },
    { 1877669, 0x23bd92a21ec515ad, 18 },
    { 2065501, 0x40fb10046a9018ad, 19 },
    { 2097091, 0x20003d007448ddab, 18 },
    { 2097169, 0xffff7800483fd99f, 21 },
    { 2272073, 0xec4a8db5565015c9, 21 },
    { 2499337, 0xd6ce2a2f0e099c3f, 21 },
    { 2749277, 0xc346f1c005a7cbfd, 21 },
    { 3024209, 0x58c31fcc0d9e6e27, 20 },
    { 3326629, 0x142c5909f109e211, 18 },
    { 3659309, 0x92b6b7b6f5977563, 21 },
    { 4025269, 0x85600abb373d0a35, 21 },
    { 4194217, 0x8000ae00ec89418b, 21 },
    { 4194319, 0xffffc4000e0ffcb5, 22 },
    { 4427809, 0xf27fe47e0a1ecbef, 22 },
    { 4870589, 0xdc7446c0edbc0001, 22 },
    { 5357657, 0xc8699e606404a9d1, 22 },
    { 5893423, 0x16c62f0323d86a9d, 19 },
    { 6482783, 0x52d09c22a2a6c2c5, 21 },
    { 7131139, 0x969224b9f2ee14a7, 22 },
    { 7844257, 0x22387b89d0b6e6c9, 20 },
    { 8388593, 0x80000f0001c20035, 22 },
    { 8388617, 0x7ffff70000a1fff5, 22 },
    { 8628709, 0x3e381a0144347401, 21 },
    { 9491579, 0x71202ffc15aa8ed7, 22 },
    { 10440743, 0x66d76d80be20283b, 22 },
    { 11484859, 0xbafbe06bc10df241, 23 },
    { 12633353, 0x153f8727ae48a69f, 20 },
    { 13896689, 0x26a20ce1ae196f1b, 21 },
    { 15286367, 0x463de6229adc3ac1, 22 },
    { 16777199, 0x4000044000484005, 22 },
    { 16777259, 0x7fffea80039c7f65, 23 },
    { 16815031, 0x3fdb2782a6bbdcf, 18 },
    { 18496567, 0x1d0682f07cd39653, 21 },
    { 20346247, 0x698c02c475e2b363, 23 },
    { 22380871, 0xbfe74b3e43622dad, 24 },
    { 24618959, 0x573a965b2a4ae29d, 23 },
    { 27080957, 0x9e98ea30217d46f9, 24 },
    { 29789063, 0x902de8e7602b49bf, 24 },
    { 32768033, 0x831265f0f6332b25, 24 },
    { 33554383, 0x80000c40012c201d, 24 },
    { 33554467, 0x3ffffba0004c8ffb, 23 },
    { 36044849, 0xee4ff9a9bf66a315, 25 },
    { 39649343, 0xd8a5c86f5f11996f, 25 },
    { 43614287, 0x6279e54ed03309b7, 24 },
    { 47975777, 0xb30c1d911abaa2c3, 25 },
    { 52773367, 0xa2c52faa5760812d, 25 },
    { 58050791, 0x49fc82bce4a6e201, 24 },
    { 63855907, 0x868545b3a2bff3a7, 25 },
    { 67108837, 0x800003600016c801, 25 },
    { 67108879, 0xfffffc40000e1, 14 },
    { 70241497, 0xf495391269a38bdd, 26 },
    { 77265649, 0x6f2c8e16aa6631f, 21 },
    { 84992227, 0x65113a5512bbb03b, 25 },
    { 93491471, 0x16f84718a0a70eb7, 23 },
    { 102840697, 0xa70d9f92afdfc6e7, 26 },
    { 113124779, 0x97ddd5cd693b9009, 26 },
    { 124437259, 0x8a0f7c651a40d4a3, 26 },
    { 134217649, 0x4000027800186101, 25 },
    { 134217757, 0x3fffff18000349, 17 },
    { 136880987, 0xfb04e1eb937502d5, 27 },
    { 150569087, 0x72197de6304ec18d, 26 },
    { 165625997, 0x67ba154f3d5602cf, 26 },
    { 182188649, 0xbc982332517906c3, 27 },
    { 200407583, 0xab7304d99725f065, 27 },
    { 220448351, 0x9bdcecafd6e80fd1, 27 },
    { 242493193, 0x8db1911664b43a9b, 27 },
    { 266742517, 0x2033fe0734c100cf, 25 },
    { 268435337, 0x800003b8001ba881, 27 },
    { 268435459, 0xffffffd0000009, 20 },
    { 293416793, 0x3a8d135ea855b9a7, 26 },
    { 322758509, 0xd4e9b93666870913, 28 },
    { 355034363, 0xc18ea843a562de51, 28 },
    { 390537803, 0xaff60d38ccdc017f, 28 },
    { 429591611, 0x9ff6f4123971ced9, 28 },
    { 472550777, 0x48b611cd16c821a9, 27 },
    { 519805879, 0x8433c2dea8227133, 28 },
    { 536870701, 0x8000034c0015bd21, 28 },
    { 536870923, 0x3fffffea0000079, 23 },
    { 571786469, 0xf05e1c6ebf337c69, 29 },
    { 628965121, 0x6d420cdda86fc957, 28 },
    { 691861657, 0xc6a6a2943f78c557, 29 },
    { 761047853, 0xb4977c0e248c0475, 29 },
    { 837152663, 0xa42c9f01685bab0f, 29 },
    { 920867963, 0x254ff580af07aa65, 27 },
    { 1012954807, 0x43d73285d40732c9, 28 },
    { 1073741399, 0x8000035200160c89, 29 },
    { 1073741827, 0xfffffff40000009, 26 },
    { 1114250327, 0x3dac5c552ae6f357, 28 },
    { 1225675387, 0x70221c13db4a31a9, 29 },
    { 1348242989, 0x65f0764d64274f17, 29 },
    { 1483067303, 0x2e560732891567fb, 28 },
    { 1631374093, 0x543f52b3cb63209d, 29 },
    { 1794511519, 0x992d5074cd2f728f, 30 },
    { 1973962681, 0x8b4077a40c3268ff, 30 },
    { 2147482819, 0x8000033d0014f913, 30 },
    { 2147483659, 0x3ffffffa80000079, 29 },
    { 2171358967, 0x7e97b283a02dce23, 30 },
    { 2388494903, 0x73158ae386eedb3d, 30 },
    { 2627344409, 0x689f3867435abcbf, 30 },
    { 2890078963, 0x2f8e30c9e58f09d5, 29 },
    { 3179086913, 0x5676e43f4f035ad3, 30 },
    { 3496995713, 0x13a6a832af8f8019, 28 },
    { 3846695299, 0x11dd5315a8ef2397, 28 },
    { 4231364953, 0x207b22a309206bb1, 29 },
    { 4294965229, 0x40000204c0104c5b, 30 },
    { 4294968929, 0x3ffffe67c00a2c3, 26 },
    { 4654501529, 0x3b0e6d74573e993b, 30 },
    { 5119951699, 0x35b00666c18bf943, 30 },
    { 5631946871, 0x186748ba30e4ed3d, 29 },
    { 6195141889, 0xb17aca7c872f421, 28 },
    { 6814656529, 0x142b0b493509f1fb, 29 },
    { 7496122223, 0x24ab5a532f77d38d, 30 },
    { 8245735111, 0x2155f4d8a714d775, 30 },
    { 8589931451, 0x200000c45004b455, 30 },
    { 8589936671, 0xfffffbf080107cf, 29 },
    { 9070308851, 0x1e4e245b08f7bce9, 30 },
    { 9977339753, 0x3719b675682d78f, 27 },
    { 10975073789, 0xc85d8023fa93b0f, 29 },
    { 12072581177, 0xb626747b7ed32d1, 29 },
    { 13279839343, 0xa5975294b5a7e51, 29 },
    { 14607823453, 0x25a2643179dc15d, 27 },
    { 16068606011, 0x111b44ccdf34fd45, 30 },
    { 17179868711, 0x8000003b20001b5, 29 },
    { 17179869479, 0x3fffffed9000055, 28 },
    { 17675468017, 0xf8d2731254dab6d, 30 },
    { 19443015083, 0xe233ae3a2af2541, 30 },
    { 21387318229, 0x3368d5e2728bc27, 28 },
    { 23526051433, 0x2ebc652b779151, 24 },
    { 25878657319, 0x2a7cb8fbc9112ff, 28 },
    { 28466523119, 0x134ff6fd4f254bd, 27 },
    { 31313175461, 0x8c741b8d130187f, 30 },
    { 34359737299, 0x400000216800117, 29 },
    { 34359743651, 0x1fffffad7400d4f, 28 },
    { 34444493077, 0x7faf5ed8fe3187f, 30 },
    { 37888942411, 0xe827952abcf5e7, 27 },
    { 41677836677, 0x34c32d892f31c49, 29 },
    { 45845621273, 0x5fee8135ed8b523, 30 },
    { 50430183631, 0x5735e9cd3db6bfb, 30 },
    { 55473203623, 0x4f4848c212096bb, 30 },
    { 61020524111, 0x48132adc7b71a57, 30 },
    { 67122576643, 0x4185c9ddbb27df5, 30 },
    { 68719476109, 0x80000013980003, 27 },
    { 68719479853, 0x3fffffcf4c00251, 30 },
    { 73834836017, 0x1dc872fc20260a9, 29 },
    { 81218319671, 0xd89a8a10071b81, 28 },
    { 89340153359, 0x313a65224777205, 30 },
    { 98274170143, 0xb302e3f34e4dff, 28 },
    { 108101589851, 0x28af33bd8fa1ff9, 30 },
    { 118911750571, 0x24fc5d8c00943d7, 30 },
    { 130802929363, 0x433f359884471b, 27 },
    { 137438942467, 0x800000abf400e7, 28 },
    { 137438954281, 0x1ffffffcd700005, 30 },
    { 143883226261, 0x3d22309cf4e3c9, 27 },
    { 158271549187, 0xde4dc805238409, 29 },
    { 174098706571, 0x194305406690e17, 30 },
    { 191508577957, 0xb7b8e05445e3d5, 29 },
    { 210659440757, 0x538294621d29c7, 28 },
    { 231725385193, 0x4beb1285c26f29, 28 },
    { 254897924539, 0x11410fd7a492689, 30 },
    { 274877905921, 0x10000000ffc0001, 30 },
    { 274877919317, 0x7fffff9f560049, 29 },
    { 280387717829, 0x7d7c161a08ab73, 29 },
    { 308426491729, 0x7213b6f341cfab, 29 },
    { 339269157187, 0x67b4d489a0602b, 29 },
    { 373196077151, 0xbc8e99906cfc85, 30 },
    { 410515695151, 0x55b52e6370d30f, 29 },
    { 451567272113, 0x9bd50e5b24029f, 30 },
    { 496724003821, 0x8daa6a0ecb29d5, 30 },
    { 546396456911, 0x20325db7fb659d, 28 },
    { 549755809793, 0x40000007ff8001, 29 },
    { 549755832433, 0x7fffffb78f0029, 30 },
    { 601036106599, 0x3a8a1ebc58600b, 29 },
    { 661139718661, 0x6a6f7dafbff1c1, 30 },
    { 727253701651, 0x60c27229d9e529, 30 },
    { 799979077199, 0x57f6964aa967c9, 30 },
    { 879976997857, 0x4ff771476beb27, 30 },
    { 967974740311, 0x24593362ac9567, 29 },
    { 1064772215531, 0x42168bf7fcda35, 30 },
    { 1099511616193, 0x20000005a7e001, 29 },
    { 1099511647841, 0x1ffffff633e003, 29 },
    { 1171249506337, 0x3c147f02ef6f75, 30 },
    { 1288374478813, 0x1b4f226dee291d, 29 },
    { 1417211952407, 0x31a72744654df1, 30 },
    { 1558933231129, 0x1691cbf38a0acf, 29 },
    { 1714826630987, 0x2909159bfcbfcd, 30 },
    { 1886309333413, 0x254e139807d82b, 30 },
    { 2074940423807, 0x21e9e3195357d, 26 },
    { 2199023209213, 0x8000002d40c01, 28 },
    { 2199023321087, 0x3fffffe000201, 27 },
    { 2282434703201, 0xf6a4fd961f339, 29 },
    { 2510678232779, 0xe038e62fafc03, 29 },
    { 2761746284051, 0x197ad43395483f, 30 },
    { 3037920919643, 0x1729d82df971c1, 30 },
    { 3341713256461, 0xa8762367ff3c5, 29 },
    { 3675884823899, 0x264964f9051f5, 27 },
    { 4043473476923, 0x22ce5bb2776d3, 27 },
    { 4398046128967, 0x800000ba97211, 29 },
    { 4398046576633, 0xffffffc001c01, 30 },
    { 4447820844161, 0x1fa4535a175c5, 27 },
    { 4892603064841, 0xe61f750c689c5, 30 },
    { 5381863486523, 0xd133de7a55bc3, 30 },
    { 5920049850509, 0xbe2f274fad243, 30 },
    { 6512054985383, 0xace50c343dfd9, 30 },
    { 7163260494679, 0x13a5aa1cb6dd, 23 },
    { 7879586622553, 0x8ee360b8fa2df, 30 },
    { 8667545651987, 0x81e5fa7a62799, 30 },
    { 8796092760067, 0x4000001fffe81, 29 },
    { 8796093207563, 0x7fffffd2bf501, 30 },
    { 9534300382801, 0x3b0b71c954c93, 29 },
    { 10487730500699, 0x6b5aa0491c629, 30 },
    { 11536504133657, 0x18660d1334e7, 24 },
    { 12690155843149, 0x58b8e9222cb63, 30 },
    { 13959172871737, 0x50a81935f850d, 30 },
    { 15355090526597, 0x24a97fc3fc28d, 29 },
    { 16890600126481, 0x42a88b118f063, 30 },
    { 17592185520139, 0x4000001fffd41, 30 },
    { 17592186568699, 0x3fffffe000141, 30 },
    { 18579660513737, 0x1e4c9c438cd87, 29 },
    { 20437626746717, 0x1b8b76c4f0717, 29 },
    { 22481390583623, 0xc8535f191473, 28 },
    { 24729530051827, 0x2d874fbedbe89, 30 },
    { 27202483472023, 0x14b1de68bad23, 29 },
    { 29922731834477, 0x25a07d1b2fd95, 30 },
    { 32915005914277, 0x2234cec344d8d, 30 },
    { 35184368165423, 0x2000003bddd17, 30 },
    { 35184375054643, 0x7fffff4afb35, 28 },
    { 36206507518087, 0x7c62efa2fd7f, 28 },
    { 39827161337543, 0xe2283e32a3d9, 29 },
    { 43809878171641, 0xcd98f282880f, 29 },
    { 48190866049457, 0x2eba08910b11, 27 },
    { 53009952991367, 0xa9ea4da037d9, 29 },
    { 58310952880457, 0x9a77e8ae62d1, 29 },
    { 64142052580913, 0x8c6d0170d3d, 25 },
    { 70368739983379, 0x1000000ffffb5, 30 },
    { 70368748371967, 0xffffff000005, 30 },
    { 70556258333689, 0xff51d3f4ed0f, 30 },
    { 77611886220161, 0x740debc775ab, 29 },
    { 85373080112477, 0x698104769a13, 29 },
    { 93910388894053, 0xbfd34dd4836f, 30 },
    { 103301427956821, 0xae6300ead457, 30 },
    { 113631571310639, 0x9e888c6b5385, 30 },
    { 124994739606907, 0x901f0a724fdb, 30 },
    { 137494215902699, 0x8304f2140afb, 30 },
    { 140737476492083, 0x4000005a8267, 29 },
    { 140737505132519, 0x7fffff00001b, 30 },
    { 151243644844067, 0x3b8de235a359, 29 },
    { 166368012741121, 0x6c47c9b09f8d, 30 },
    { 183004819085009, 0x626fce72d0b7, 30 },
    { 201305307931187, 0x597cea06170f, 30 },
    { 221435852191409, 0x515a48c9cd31, 30 },
    { 243579444047761, 0x127d3f0e2387, 28 },
    { 267937395114787, 0x219de6f4691, 25 },
    { 281474959933529, 0x2000001ffff5, 29 },
    { 281474993487841, 0x7fffff80001, 27 },
    { 294731135770343, 0x1e8f8c2204e9, 29 },
    { 324204257711149, 0x3790a19a2ad3, 30 },
    { 356624688336731, 0x32837b97e383, 30 },
    { 392287159075237, 0x16f5f25a714b, 29 },
    { 431515897820233, 0x29bf2cdc76df, 30 },
    { 474667499178151, 0x97ce7457f0b, 28 },
    { 522134278298861, 0x114030111429, 29 },
    { 562949929695089, 0x20000016a089, 30 },
    { 562949977147729, 0x1fffffe95f6b, 30 },
    { 574347706497419, 0xfaeb7552ce9, 29 },
    { 631782506381069, 0x3906f795181, 27 },
    { 694960789199221, 0x19ebe4d28aa5, 30 },
    { 764456910904769, 0x1790a16374af, 30 },
    { 840902621099003, 0x156c3598113d, 30 },
    { 924992898489329, 0x1379a5107b59, 30 },
    { 1017492211448891, 0x11b4677c99b7, 30 },
    { 1119241449141419, 0x40617840c27, 28 },
    { 0, 0, 0 }
};

} // namespace detail

void hash_prime::_check()
{
    if (_prime == 0)
    {
        _muldiv = 0;
        _shift = 0;
        return;
    }
    // basically _muldiv must satisfy the equation
    //   _muldiv = ((uint128_t(1)<<(64+_shift)) / _prime) + 1
    // but must also pass the following tests to ensure correctness for all uint64_t input values
    uint128_t bigint = uint128_t(_muldiv) * _prime;
    if (uint64_t(bigint>>64) != (uint64_t(1)<<_shift))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 1)");
    if (uint64_t(bigint) >= _prime)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 2)");
    // check validity by checking correct results for 6 specific values
    if (mod(1) != 1)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 3)");
    if (mod(_prime-1) != _prime-1)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 4)");
    if (mod(_prime) != 0)
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 5)");
    uint64_t maxint = ~uint64_t(0);
    if (mod(maxint) != (maxint % _prime))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 6)");
    maxint -= (maxint % _prime);
    if (mod(maxint) != (maxint % _prime))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 7)");
    --maxint;
    if (mod(maxint) != (maxint % _prime))
        throw std::runtime_error("hash_prime::_check(): invalid parameters (fail 8)");
}

// obtain smallest internal hash_prime with prime > n
hash_prime get_hash_prime_gt(uint64_t n)
{
    auto it = detail::hash_prime_table + 0;
    for (; it->prime != 0; ++it)
        if (it->prime > n)
            return hash_prime(it->prime, it->muldiv, it->shift);
    throw std::runtime_error("get_hash_prime_gt(): could not find suitable hash_prime");
}

// obtain smallest internal hash_prime with prime >= n
hash_prime get_hash_prime_ge(uint64_t n)
{
    auto it = detail::hash_prime_table + 0;
    for (; it->prime != 0; ++it)
        if (it->prime >= n)
            return hash_prime(it->prime, it->muldiv, it->shift);
    throw std::runtime_error("get_hash_prime_ge(): could not find suitable hash_prime");
}

// obtain largest internal hash_prime with prime < n
hash_prime get_hash_prime_lt(uint64_t n)
{
    auto begin = detail::hash_prime_table + 0;
    auto it = begin;
    for (; it->prime != 0 && it->prime < n; ++it)
        ;
    if (it == begin)
        throw std::runtime_error("get_hash_prime_lt(): could not find suitable hash_prime");
    --it;
    return hash_prime(it->prime, it->muldiv, it->shift);
}

// obtain largest internal hash_prime with prime <= n
hash_prime get_hash_prime_le(uint64_t n)
{
    auto begin = detail::hash_prime_table + 0;
    auto it = begin;
    for (; it->prime != 0 && it->prime <= n; ++it)
        ;
    if (it == begin)
        throw std::runtime_error("get_hash_prime_lt(): could not find suitable hash_prime");
    --it;
    return hash_prime(it->prime, it->muldiv, it->shift);
}

MCCL_END_NAMESPACE
