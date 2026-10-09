# GP / Kernel Exploration: CheMeleon vs Descriptor Embeddings

timestamp: 2026-10-08 19:16:10.295512

Inductive GP (test points independent). Kernels: RBF, Matern-5/2,
Spectral Mixture (MLE), Tanimoto(FP), Tanimoto(FP)xRBF(desc),
Heteroscedastic RBF (IRWLS linear log-noise).
Spaces: CheMeleon 2048 (raw+white) vs descriptor 2373 (Morgan count + RDKit).
HP: grid on <=1200-pt subsample (20% val), refit full train, score test.


## `polaris/pkis2-ret-wt-reg-v2`  (n_tr=534, mean_squared_error)

|                               | 0                           |
|:------------------------------|:----------------------------|
| bench                         | polaris/pkis2-ret-wt-reg-v2 |
| metric                        | mean_squared_error          |
| n_train                       | 534                         |
| n_test                        | 106                         |
| chem/raw/rbf                  | 810.1035468598585           |
| chem/raw/rbf_val              | 0.650466194699206           |
| chem/raw/matern52             | 846.5525299981422           |
| chem/raw/matern52_val         | 0.654784301604939           |
| chem/raw/smk                  | 810.9403421334546           |
| chem/raw/smk_val              | 0.6461289042727324          |
| chem/raw/rbf_heto_hetoscale   | 4.6450153978217205e-05      |
| chem/raw/rbf_heto             | 806.9443155279889           |
| chem/raw/rbf_heto_val         | 0.6500287722315572          |
| chem/raw/rank_kept            | 409                         |
| chem/white/rbf                | 960.6462038752491           |
| chem/white/rbf_val            | 0.9140731350845355          |
| chem/white/matern52           | 896.8185066919178           |
| chem/white/matern52_val       | 0.9143236950241891          |
| chem/white/smk                | 896.842396357606            |
| chem/white/smk_val            | 0.9231708070830015          |
| chem/white/rbf_heto_hetoscale | 0.040597161859965335        |
| chem/white/rbf_heto           | 935.1118899554398           |
| chem/white/rbf_heto_val       | 0.9313723751908107          |
| chem/white/rank_kept          | 409                         |
| desc/rbf                      | 853.8219730565952           |
| desc/rbf_val                  | 0.6194277280740615          |
| desc/matern52                 | 874.9336692272959           |
| desc/matern52_val             | 0.6217327297225106          |
| desc/smk                      | 852.5969378117808           |
| desc/smk_val                  | 0.6347940512478745          |
| desc/tanimoto_fp              | 850.7085270326704           |
| desc/tanimoto_fp_val          | 0.6357523219619263          |
| desc/tanimoto_x_rbf           | 886.9699295130745           |
| desc/tanimoto_x_rbf_val       | 0.6736096715205808          |
| desc/rbf_heto_hetoscale       | 9.968962159065611e-06       |
| desc/rbf_heto                 | 869.0548978768416           |
| desc/rbf_heto_val             | 0.6181317520113133          |

## `polaris/pkis2-kit-wt-reg-v2`  (n_tr=524, mean_squared_error)

|                               | 0                           |
|:------------------------------|:----------------------------|
| bench                         | polaris/pkis2-kit-wt-reg-v2 |
| metric                        | mean_squared_error          |
| n_train                       | 524                         |
| n_test                        | 116                         |
| chem/raw/rbf                  | 939.7103687031267           |
| chem/raw/rbf_val              | 0.34941414714436453         |
| chem/raw/matern52             | 934.6381078834322           |
| chem/raw/matern52_val         | 0.34519473383253946         |
| chem/raw/smk                  | 991.8643086091434           |
| chem/raw/smk_val              | 0.3543347586353438          |
| chem/raw/rbf_heto_hetoscale   | 1.0834330888685718e-06      |
| chem/raw/rbf_heto             | 942.1276883822059           |
| chem/raw/rbf_heto_val         | 0.35294628620288887         |
| chem/raw/rank_kept            | 400                         |
| chem/white/rbf                | 1093.7301074701782          |
| chem/white/rbf_val            | 0.6041974350199688          |
| chem/white/matern52           | 1090.256185230558           |
| chem/white/matern52_val       | 0.6210578657248867          |
| chem/white/smk                | 1095.418611421812           |
| chem/white/smk_val            | 0.6208590307131184          |
| chem/white/rbf_heto_hetoscale | 0.0068519859774098454       |
| chem/white/rbf_heto           | 1088.5801700263128          |
| chem/white/rbf_heto_val       | 0.5993059405845926          |
| chem/white/rank_kept          | 400                         |
| desc/rbf                      | 956.6554764829195           |
| desc/rbf_val                  | 0.33939676164578825         |
| desc/matern52                 | 963.8960921749901           |
| desc/matern52_val             | 0.3432915005467074          |
| desc/smk                      | 1051.5558813508133          |
| desc/smk_val                  | 0.3776413710670228          |
| desc/tanimoto_fp              | 937.8110574131667           |
| desc/tanimoto_fp_val          | 0.34025753850377183         |
| desc/tanimoto_x_rbf           | 955.8394401668038           |
| desc/tanimoto_x_rbf_val       | 0.3605019689229403          |
| desc/rbf_heto_hetoscale       | 0.0001815669132519827       |
| desc/rbf_heto                 | 984.793719292445            |
| desc/rbf_heto_val             | 0.36498446286955866         |

## `polaris/pkis2-egfr-wt-reg-v2`  (n_tr=496, mean_squared_error)

|                               | 0                            |
|:------------------------------|:-----------------------------|
| bench                         | polaris/pkis2-egfr-wt-reg-v2 |
| metric                        | mean_squared_error           |
| n_train                       | 496                          |
| n_test                        | 144                          |
| chem/raw/rbf                  | 516.2031202471187            |
| chem/raw/rbf_val              | 0.4239987111851435           |
| chem/raw/matern52             | 524.5858374960217            |
| chem/raw/matern52_val         | 0.4200418400001816           |
| chem/raw/smk                  | 521.5344003321924            |
| chem/raw/smk_val              | 0.4480484757905508           |
| chem/raw/rbf_heto_hetoscale   | 1.2381773486901955e-05       |
| chem/raw/rbf_heto             | 534.5985994836086            |
| chem/raw/rbf_heto_val         | 0.41318481324383116          |
| chem/raw/rank_kept            | 395                          |
| chem/white/rbf                | 564.3926737051456            |
| chem/white/rbf_val            | 0.7212605487039896           |
| chem/white/matern52           | 559.916688839319             |
| chem/white/matern52_val       | 0.7185980724719628           |
| chem/white/smk                | 564.833388885279             |
| chem/white/smk_val            | 0.738446155139255            |
| chem/white/rbf_heto_hetoscale | 1.1874349121457473e-06       |
| chem/white/rbf_heto           | 552.8596383331494            |
| chem/white/rbf_heto_val       | 0.7338228543844294           |
| chem/white/rank_kept          | 395                          |
| desc/rbf                      | 542.8501896342937            |
| desc/rbf_val                  | 0.41023740038996787          |
| desc/matern52                 | 545.9913715422032            |
| desc/matern52_val             | 0.41883608740768186          |
| desc/smk                      | 602.4698972829909            |
| desc/smk_val                  | 0.5713984467491002           |
| desc/tanimoto_fp              | 549.9796613557143            |
| desc/tanimoto_fp_val          | 0.4580988781374236           |
| desc/tanimoto_x_rbf           | 564.9346927261231            |
| desc/tanimoto_x_rbf_val       | 0.4870792801060642           |
| desc/rbf_heto_hetoscale       | 1.0841313855747932e-05       |
| desc/rbf_heto                 | 565.2987053828094            |
| desc/rbf_heto_val             | 0.4464283117688751           |

## `polaris/adme-fang-solu-1`  (n_tr=1578, pearsonr)

|                               | 0                        |
|:------------------------------|:-------------------------|
| bench                         | polaris/adme-fang-solu-1 |
| metric                        | pearsonr                 |
| n_train                       | 1578                     |
| n_test                        | 400                      |
| chem/raw/rbf                  | 0.6310806756828573       |
| chem/raw/rbf_val              | 0.5519905370522782       |
| chem/raw/matern52             | 0.6375054253728588       |
| chem/raw/matern52_val         | 0.5525866140124427       |
| chem/raw/smk                  | 0.6408914358091838       |
| chem/raw/smk_val              | 0.5521406820090616       |
| chem/raw/rbf_heto_hetoscale   | 1.0109326332012805e-06   |
| chem/raw/rbf_heto             | 0.6284467119906986       |
| chem/raw/rbf_heto_val         | 0.5437637559962498       |
| chem/raw/rank_kept            | 1080                     |
| chem/white/rbf                | 0.5753127526381294       |
| chem/white/rbf_val            | 0.36649293768290864      |
| chem/white/matern52           | 0.5839982347260343       |
| chem/white/matern52_val       | 0.369828259520444        |
| chem/white/smk                | 0.5710358169242871       |
| chem/white/smk_val            | 0.3634722953544719       |
| chem/white/rbf_heto_hetoscale | 0.05788818679322904      |
| chem/white/rbf_heto           | 0.5305338023759565       |
| chem/white/rbf_heto_val       | 0.36733888017416827      |
| chem/white/rank_kept          | 1080                     |
| desc/rbf                      | 0.5676007919988175       |
| desc/rbf_val                  | 0.5632819924469303       |
| desc/matern52                 | 0.5880278878862113       |
| desc/matern52_val             | 0.565895924024915        |
| desc/smk                      | 0.5773095817673843       |
| desc/smk_val                  | 0.5611485358202614       |
| desc/tanimoto_fp              | 0.5862507364630352       |
| desc/tanimoto_fp_val          | 0.5650983604941803       |
| desc/tanimoto_x_rbf           | 0.5852809305732964       |
| desc/tanimoto_x_rbf_val       | 0.5635037748932358       |
| desc/rbf_heto_hetoscale       | 5.73677316108132e-05     |
| desc/rbf_heto                 | 0.5179828603494361       |
| desc/rbf_heto_val             | 0.5665958483382635       |

## `polaris/adme-fang-rppb-1`  (n_tr=111, pearsonr)

|                               | 0                        |
|:------------------------------|:-------------------------|
| bench                         | polaris/adme-fang-rppb-1 |
| metric                        | pearsonr                 |
| n_train                       | 111                      |
| n_test                        | 24                       |
| chem/raw/rbf                  | 0.5427483745702747       |
| chem/raw/rbf_val              | 0.584170383979388        |
| chem/raw/matern52             | 0.5375490833829256       |
| chem/raw/matern52_val         | 0.60166809917732         |
| chem/raw/smk                  | 0.5380933570984511       |
| chem/raw/smk_val              | 0.43391896596692564      |
| chem/raw/rbf_heto_hetoscale   | 0.0021776911080183493    |
| chem/raw/rbf_heto             | 0.5084349334828179       |
| chem/raw/rbf_heto_val         | 0.5720673320100889       |
| chem/raw/rank_kept            | 105                      |
| chem/white/rbf                | 0.7883914002028769       |
| chem/white/rbf_val            | 0.32861171925073984      |
| chem/white/matern52           | 0.7957959744105549       |
| chem/white/matern52_val       | 0.33146839915932247      |
| chem/white/smk                | 0.0012626088535922907    |
| chem/white/smk_val            | -0.033718626002915385    |
| chem/white/rbf_heto_hetoscale | 1.000000000000004e-06    |
| chem/white/rbf_heto           | 0.7948219639117924       |
| chem/white/rbf_heto_val       | 0.3167892879672578       |
| chem/white/rank_kept          | 105                      |
| desc/rbf                      | -0.12442901181632154     |
| desc/rbf_val                  | 0.5177770994084422       |
| desc/matern52                 | -0.12213600401950468     |
| desc/matern52_val             | 0.5198552659486023       |
| desc/smk                      | -0.09155541552603572     |
| desc/smk_val                  | 0.40338636337203204      |
| desc/tanimoto_fp              | -0.12086481761668053     |
| desc/tanimoto_fp_val          | 0.5180055960790836       |
| desc/tanimoto_x_rbf           | -0.07501542068921703     |
| desc/tanimoto_x_rbf_val       | 0.32468287145742425      |
| desc/rbf_heto_hetoscale       | 0.0015761685182466977    |
| desc/rbf_heto                 | -0.1252746572629089      |
| desc/rbf_heto_val             | 0.5209341820009046       |

## `polaris/adme-fang-hppb-1`  (n_tr=126, pearsonr)

|                               | 0                        |
|:------------------------------|:-------------------------|
| bench                         | polaris/adme-fang-hppb-1 |
| metric                        | pearsonr                 |
| n_train                       | 126                      |
| n_test                        | 34                       |
| chem/raw/rbf                  | 0.6530244285150116       |
| chem/raw/rbf_val              | 0.3766105442707224       |
| chem/raw/matern52             | 0.6441624843023326       |
| chem/raw/matern52_val         | 0.4045052585311891       |
| chem/raw/smk                  | 0.6865581793591343       |
| chem/raw/smk_val              | 0.3342932200038868       |
| chem/raw/rbf_heto_hetoscale   | 2.0788398599683477e-05   |
| chem/raw/rbf_heto             | 0.6988073748849196       |
| chem/raw/rbf_heto_val         | 0.43797817291613556      |
| chem/raw/rank_kept            | 119                      |
| chem/white/rbf                | 0.09528690418861673      |
| chem/white/rbf_val            | 0.14442179155766616      |
| chem/white/matern52           | 0.28144063002831377      |
| chem/white/matern52_val       | 0.3295460020336086       |
| chem/white/smk                | 0.7832265598597442       |
| chem/white/smk_val            | -0.12544102017800823     |
| chem/white/rbf_heto_hetoscale | 1.000000000000004e-06    |
| chem/white/rbf_heto           | 0.09556789373456046      |
| chem/white/rbf_heto_val       | 0.14442179155766618      |
| chem/white/rank_kept          | 119                      |
| desc/rbf                      | 0.30101991964517427      |
| desc/rbf_val                  | 0.5148682406360507       |
| desc/matern52                 | 0.33480072641481845      |
| desc/matern52_val             | 0.4204455241978766       |
| desc/smk                      | 0.3690924443268807       |
| desc/smk_val                  | 0.36278273838577824      |
| desc/tanimoto_fp              | 0.3627682332949823       |
| desc/tanimoto_fp_val          | 0.3200218034343039       |
| desc/tanimoto_x_rbf           | 0.3343757545609255       |
| desc/tanimoto_x_rbf_val       | 0.40282698363308406      |
| desc/rbf_heto_hetoscale       | 1.000000000000004e-06    |
| desc/rbf_heto                 | 0.30102874261508866      |
| desc/rbf_heto_val             | 0.5148714385965618       |

## `polaris/adme-fang-perm-1`  (n_tr=1919, pearsonr)

|                               | 0                        |
|:------------------------------|:-------------------------|
| bench                         | polaris/adme-fang-perm-1 |
| metric                        | pearsonr                 |
| n_train                       | 1919                     |
| n_test                        | 483                      |
| chem/raw/rbf                  | 0.7593591336042438       |
| chem/raw/rbf_val              | 0.7520030836258802       |
| chem/raw/matern52             | 0.7524937031725637       |
| chem/raw/matern52_val         | 0.748190365889333        |
| chem/raw/smk                  | 0.76763191095111         |
| chem/raw/smk_val              | 0.7512440243286721       |
| chem/raw/rbf_heto_hetoscale   | 9.47282691839715e-06     |
| chem/raw/rbf_heto             | 0.7495126238067592       |
| chem/raw/rbf_heto_val         | 0.7374247273589417       |
| chem/raw/rank_kept            | 1174                     |
| chem/white/rbf                | 0.7068107004777873       |
| chem/white/rbf_val            | 0.4288780730175237       |
| chem/white/matern52           | 0.7147998336178686       |
| chem/white/matern52_val       | 0.43226313431303287      |
| chem/white/smk                | 0.7172599886570271       |
| chem/white/smk_val            | 0.4321231123966264       |
| chem/white/rbf_heto_hetoscale | 1.873041887023197e-05    |
| chem/white/rbf_heto           | 0.6908671318499955       |
| chem/white/rbf_heto_val       | 0.42573433652915216      |
| chem/white/rank_kept          | 1174                     |
| desc/rbf                      | 0.7531106986048144       |
| desc/rbf_val                  | 0.6997015471013398       |
| desc/matern52                 | 0.7604041802741088       |
| desc/matern52_val             | 0.6954894508925313       |
| desc/smk                      | 0.7736104948722246       |
| desc/smk_val                  | 0.6943072624563551       |
| desc/tanimoto_fp              | 0.7681537169823728       |
| desc/tanimoto_fp_val          | 0.7022744498540985       |
| desc/tanimoto_x_rbf           | 0.7199850922243537       |
| desc/tanimoto_x_rbf_val       | 0.5978168842508476       |
| desc/rbf_heto_hetoscale       | 6.225975593038498e-05    |
| desc/rbf_heto                 | 0.7584669021110689       |
| desc/rbf_heto_val             | 0.703870266729493        |

## `polaris/adme-fang-rclint-1`  (n_tr=2218, pearsonr)

|                               | 0                          |
|:------------------------------|:---------------------------|
| bench                         | polaris/adme-fang-rclint-1 |
| metric                        | pearsonr                   |
| n_train                       | 2218                       |
| n_test                        | 559                        |
| chem/raw/rbf                  | 0.7123660598265338         |
| chem/raw/rbf_val              | 0.6920958896433647         |
| chem/raw/matern52             | 0.7164146624000733         |
| chem/raw/matern52_val         | 0.6953518580783998         |
| chem/raw/smk                  | 0.7219116076383404         |
| chem/raw/smk_val              | 0.6871809134763662         |
| chem/raw/rbf_heto_hetoscale   | 3.565444140966426e-05      |
| chem/raw/rbf_heto             | 0.7014183264430228         |
| chem/raw/rbf_heto_val         | 0.6887868646147816         |
| chem/raw/rank_kept            | 1227                       |
| chem/white/rbf                | 0.6381061921276082         |
| chem/white/rbf_val            | 0.4313220295097468         |
| chem/white/matern52           | 0.6159010328934299         |
| chem/white/matern52_val       | 0.4311161709446032         |
| chem/white/smk                | 0.6357881382095302         |
| chem/white/smk_val            | 0.426836750507168          |
| chem/white/rbf_heto_hetoscale | 0.14697256070048847        |
| chem/white/rbf_heto           | 0.6178823240275512         |
| chem/white/rbf_heto_val       | 0.43215553890925684        |
| chem/white/rank_kept          | 1227                       |
| desc/rbf                      | 0.6522281076872583         |
| desc/rbf_val                  | 0.6524277631906806         |
| desc/matern52                 | 0.6674505027970367         |
| desc/matern52_val             | 0.6562175238278205         |
| desc/smk                      | 0.6592723231514596         |
| desc/smk_val                  | 0.6575373091993122         |
| desc/tanimoto_fp              | 0.6712744135106635         |
| desc/tanimoto_fp_val          | 0.6536437611202028         |
| desc/tanimoto_x_rbf           | 0.5907777855175875         |
| desc/tanimoto_x_rbf_val       | 0.5722911912289697         |
| desc/rbf_heto_hetoscale       | 2.5659343196834717e-06     |
| desc/rbf_heto                 | 0.6346019485668956         |
| desc/rbf_heto_val             | 0.6507405708497174         |

## `polaris/adme-fang-hclint-1`  (n_tr=2229, pearsonr)

|                               | 0                          |
|:------------------------------|:---------------------------|
| bench                         | polaris/adme-fang-hclint-1 |
| metric                        | pearsonr                   |
| n_train                       | 2229                       |
| n_test                        | 575                        |
| chem/raw/rbf                  | 0.677228097338986          |
| chem/raw/rbf_val              | 0.5946295827867827         |
| chem/raw/matern52             | 0.6850504402545187         |
| chem/raw/matern52_val         | 0.6004679574047718         |
| chem/raw/smk                  | 0.6884958486251602         |
| chem/raw/smk_val              | 0.6091433077469957         |
| chem/raw/rbf_heto_hetoscale   | 1.948191508843075e-06      |
| chem/raw/rbf_heto             | 0.6753174270588069         |
| chem/raw/rbf_heto_val         | 0.5945950768248571         |
| chem/raw/rank_kept            | 1233                       |
| chem/white/rbf                | 0.6044638859096401         |
| chem/white/rbf_val            | 0.3385751574832775         |
| chem/white/matern52           | 0.5898724513862218         |
| chem/white/matern52_val       | 0.3455911755735739         |
| chem/white/smk                | 0.5910342755466118         |
| chem/white/smk_val            | 0.3413930238660616         |
| chem/white/rbf_heto_hetoscale | 4.103382247865738e-05      |
| chem/white/rbf_heto           | 0.6060811734885782         |
| chem/white/rbf_heto_val       | 0.32688902596382763        |
| chem/white/rank_kept          | 1233                       |
| desc/rbf                      | 0.6037367186312284         |
| desc/rbf_val                  | 0.6063472068244602         |
| desc/matern52                 | 0.5814540358050275         |
| desc/matern52_val             | 0.5992778759734345         |
| desc/smk                      | 0.6339896481358911         |
| desc/smk_val                  | 0.5922537482053132         |
| desc/tanimoto_fp              | 0.6343878716722234         |
| desc/tanimoto_fp_val          | 0.6129826733393995         |
| desc/tanimoto_x_rbf           | 0.5936755967385098         |
| desc/tanimoto_x_rbf_val       | 0.5449194168860215         |
| desc/rbf_heto_hetoscale       | 0.00012697354641011497     |
| desc/rbf_heto                 | 0.5918404822196472         |
| desc/rbf_heto_val             | 0.6026527987629917         |

## `tdcommons/lipophilicity-astrazeneca`  (n_tr=3360, mean_absolute_error)

|                               | 0                                   |
|:------------------------------|:------------------------------------|
| bench                         | tdcommons/lipophilicity-astrazeneca |
| metric                        | mean_absolute_error                 |
| n_train                       | 3360                                |
| n_test                        | 840                                 |
| chem/raw/rbf                  | 0.5162315831311076                  |
| chem/raw/rbf_val              | 0.5139634812842827                  |
| chem/raw/matern52             | 0.8057780894900147                  |
| chem/raw/matern52_val         | 0.5121808878325117                  |
| chem/raw/smk                  | 0.5102918873763347                  |
| chem/raw/smk_val              | 0.508313501199038                   |
| chem/raw/rbf_heto_hetoscale   | 0.0002003332439712299               |
| chem/raw/rbf_heto             | 0.5107670666746704                  |
| chem/raw/rbf_heto_val         | 0.5194625941859675                  |
| chem/raw/rank_kept            | 1283                                |
| chem/white/rbf                | 0.5662505828399497                  |
| chem/white/rbf_val            | 0.6108801828578625                  |
| chem/white/matern52           | 1.3261732421797292                  |
| chem/white/matern52_val       | 0.6238466520151243                  |
| chem/white/smk                | 0.5747882292713655                  |
| chem/white/smk_val            | 0.620143383758883                   |
| chem/white/rbf_heto_hetoscale | 0.010470007912090311                |
| chem/white/rbf_heto           | 0.5660644143096811                  |
| chem/white/rbf_heto_val       | 0.6136994401814879                  |
| chem/white/rank_kept          | 1283                                |
| desc/rbf                      | 0.5980983750703575                  |
| desc/rbf_val                  | 0.516458157702628                   |
| desc/matern52                 | 0.6051879214715881                  |
| desc/matern52_val             | 0.521883580821846                   |
| desc/smk                      | 0.5972649069006472                  |
| desc/smk_val                  | 0.5146768222397581                  |
| desc/tanimoto_fp              | 0.609646028364856                   |
| desc/tanimoto_fp_val          | 0.5279496782110461                  |
| desc/tanimoto_x_rbf           | 0.6813887100039074                  |
| desc/tanimoto_x_rbf_val       | 0.5704304672000607                  |
| desc/rbf_heto_hetoscale       | 1.1959086022146673e-06              |
| desc/rbf_heto                 | 0.6059885539322394                  |
| desc/rbf_heto_val             | 0.5165194896179887                  |

## `tdcommons/ppbr-az`  (n_tr=2231, mean_absolute_error)

|                               | 0                     |
|:------------------------------|:----------------------|
| bench                         | tdcommons/ppbr-az     |
| metric                        | mean_absolute_error   |
| n_train                       | 2231                  |
| n_test                        | 559                   |
| chem/raw/rbf                  | 8.559057828878558     |
| chem/raw/rbf_val              | 0.41046695994886134   |
| chem/raw/matern52             | 8.985073243757217     |
| chem/raw/matern52_val         | 0.41610974655255434   |
| chem/raw/smk                  | 8.973554710389424     |
| chem/raw/smk_val              | 0.43911337455930194   |
| chem/raw/rbf_heto_hetoscale   | 0.0011483349192581579 |
| chem/raw/rbf_heto             | 8.23626419545712      |
| chem/raw/rbf_heto_val         | 0.3894793543184269    |
| chem/raw/rank_kept            | 905                   |
| chem/white/rbf                | 9.749527574701576     |
| chem/white/rbf_val            | 0.4690884277452792    |
| chem/white/matern52           | 9.947346944938781     |
| chem/white/matern52_val       | 0.47869046870021503   |
| chem/white/smk                | 10.439744725276634    |
| chem/white/smk_val            | 0.5419965005546075    |
| chem/white/rbf_heto_hetoscale | 0.03455487630553368   |
| chem/white/rbf_heto           | 10.25703933272167     |
| chem/white/rbf_heto_val       | 0.4698600120257561    |
| chem/white/rank_kept          | 905                   |
| desc/rbf                      | 8.824326673439753     |
| desc/rbf_val                  | 0.40581769840322246   |
| desc/matern52                 | 8.792385024940234     |
| desc/matern52_val             | 0.4059765164097206    |
| desc/smk                      | 9.688671054080224     |
| desc/smk_val                  | 0.43397090391342524   |
| desc/tanimoto_fp              | 8.881863619469362     |
| desc/tanimoto_fp_val          | 0.40636154762906007   |
| desc/tanimoto_x_rbf           | 9.72271428642349      |
| desc/tanimoto_x_rbf_val       | 0.4275122553807131    |
| desc/rbf_heto_hetoscale       | 0.0009077901193821343 |
| desc/rbf_heto                 | 9.088788342418846     |
| desc/rbf_heto_val             | 0.3979366835220748    |

## `tdcommons/clearance-hepatocyte-az`  (n_tr=970, spearmanr)

|                               | 0                                 |
|:------------------------------|:----------------------------------|
| bench                         | tdcommons/clearance-hepatocyte-az |
| metric                        | spearmanr                         |
| n_train                       | 970                               |
| n_test                        | 243                               |
| chem/raw/rbf                  | 0.3889165580713841                |
| chem/raw/rbf_val              | 0.5475774459782191                |
| chem/raw/matern52             | 0.38271789944401513               |
| chem/raw/matern52_val         | 0.5432595353531053                |
| chem/raw/smk                  | 0.39018341570377596               |
| chem/raw/smk_val              | 0.4879486073551521                |
| chem/raw/rbf_heto_hetoscale   | 0.004753953299345491              |
| chem/raw/rbf_heto             | 0.3975632698000615                |
| chem/raw/rbf_heto_val         | 0.5486077160892278                |
| chem/raw/rank_kept            | 632                               |
| chem/white/rbf                | 0.3591020928033393                |
| chem/white/rbf_val            | 0.27034337126296737               |
| chem/white/matern52           | 0.34823669556602904               |
| chem/white/matern52_val       | 0.27621862863461755               |
| chem/white/smk                | 0.3745894635116976                |
| chem/white/smk_val            | 0.21862315284454797               |
| chem/white/rbf_heto_hetoscale | 0.03403488360142912               |
| chem/white/rbf_heto           | 0.3279244600672994                |
| chem/white/rbf_heto_val       | 0.27974015940573843               |
| chem/white/rank_kept          | 632                               |
| desc/rbf                      | 0.37288202384256663               |
| desc/rbf_val                  | 0.4547793515182282                |
| desc/matern52                 | 0.37143572788659435               |
| desc/matern52_val             | 0.4602831936192244                |
| desc/smk                      | 0.35083984111284827               |
| desc/smk_val                  | 0.3878166255882726                |
| desc/tanimoto_fp              | 0.3719130034995048                |
| desc/tanimoto_fp_val          | 0.46158935755341257               |
| desc/tanimoto_x_rbf           | 0.3850271257446316                |
| desc/tanimoto_x_rbf_val       | 0.43216443245592784               |
| desc/rbf_heto_hetoscale       | 0.0175313242303573                |
| desc/rbf_heto                 | 0.37516265434316737               |
| desc/rbf_heto_val             | 0.5210264050277491                |

## `tdcommons/half-life-obach`  (n_tr=532, spearmanr)

|                               | 0                         |
|:------------------------------|:--------------------------|
| bench                         | tdcommons/half-life-obach |
| metric                        | spearmanr                 |
| n_train                       | 532                       |
| n_test                        | 135                       |
| chem/raw/rbf                  | 0.13142207728514307       |
| chem/raw/rbf_val              | 0.3033066988619334        |
| chem/raw/matern52             | 0.2251751266650163        |
| chem/raw/matern52_val         | 0.288758660323731         |
| chem/raw/smk                  | 0.2233287491640911        |
| chem/raw/smk_val              | 0.29851950225254914       |
| chem/raw/rbf_heto_hetoscale   | 7.526906267952537e-06     |
| chem/raw/rbf_heto             | 0.34267742006933993       |
| chem/raw/rbf_heto_val         | 0.4155639397748307        |
| chem/raw/rank_kept            | 462                       |
| chem/white/rbf                | 0.14858162476997622       |
| chem/white/rbf_val            | 0.19134172889992476       |
| chem/white/matern52           | 0.10557471639204052       |
| chem/white/matern52_val       | 0.1857633850614107        |
| chem/white/smk                | 0.06253018376210563       |
| chem/white/smk_val            | 0.14350503688636218       |
| chem/white/rbf_heto_hetoscale | 1.0000000000000004e-06    |
| chem/white/rbf_heto           | 0.06145699731847626       |
| chem/white/rbf_heto_val       | 0.14702740576000384       |
| chem/white/rank_kept          | 462                       |

## `tdcommons/clearance-microsome-az`  (n_tr=881, spearmanr)

|                               | 0                                |
|:------------------------------|:---------------------------------|
| bench                         | tdcommons/clearance-microsome-az |
| metric                        | spearmanr                        |
| n_train                       | 881                              |
| n_test                        | 221                              |
| chem/raw/rbf                  | 0.5777991175977537               |
| chem/raw/rbf_val              | 0.640728509082672                |
| chem/raw/matern52             | 0.5742380515059766               |
| chem/raw/matern52_val         | 0.6439871460171039               |
| chem/raw/smk                  | 0.5792262071152592               |
| chem/raw/smk_val              | 0.6394507656809376               |
| chem/raw/rbf_heto_hetoscale   | 0.00024719589655096007           |
| chem/raw/rbf_heto             | 0.4921548621639467               |
| chem/raw/rbf_heto_val         | 0.6447763733501088               |
| chem/raw/rank_kept            | 663                              |
| chem/white/rbf                | 0.4568372469663416               |
| chem/white/rbf_val            | 0.39495685742482756              |
| chem/white/matern52           | 0.3839017443377102               |
| chem/white/matern52_val       | 0.40640288952487874              |
| chem/white/smk                | 0.44535690582211396              |
| chem/white/smk_val            | 0.35820748158176813              |
| chem/white/rbf_heto_hetoscale | 4.015123826273761e-06            |
| chem/white/rbf_heto           | 0.4145506420212979               |
| chem/white/rbf_heto_val       | 0.41887849439220687              |
| chem/white/rank_kept          | 663                              |
| desc/rbf                      | 0.5811744530231677               |
| desc/rbf_val                  | 0.5647737624240156               |
| desc/matern52                 | 0.5598283120033779               |
| desc/matern52_val             | 0.5577948017442012               |
| desc/smk                      | 0.5472009453017525               |
| desc/smk_val                  | 0.5414781414714843               |
| desc/tanimoto_fp              | 0.5683202527275891               |
| desc/tanimoto_fp_val          | 0.5482301713459163               |
| desc/tanimoto_x_rbf           | 0.5449766456930258               |
| desc/tanimoto_x_rbf_val       | 0.5413305805536986               |
| desc/rbf_heto_hetoscale       | 0.0014793121332440304            |
| desc/rbf_heto                 | 0.4957840700511948               |
| desc/rbf_heto_val             | 0.5679563831280732               |

## `tdcommons/vdss-lombardo`  (n_tr=904, spearmanr)

|                               | 0                       |
|:------------------------------|:------------------------|
| bench                         | tdcommons/vdss-lombardo |
| metric                        | spearmanr               |
| n_train                       | 904                     |
| n_test                        | 226                     |
| chem/raw/rbf                  | 0.4051738053992875      |
| chem/raw/rbf_val              | 0.5498152603471707      |
| chem/raw/matern52             | 0.4773516574331678      |
| chem/raw/matern52_val         | 0.548304818436215       |
| chem/raw/smk                  | 0.3357000915315729      |
| chem/raw/smk_val              | 0.3993544620060565      |
| chem/raw/rbf_heto_hetoscale   | 1.7118593156419076e-06  |
| chem/raw/rbf_heto             | 0.5072851408360602      |
| chem/raw/rbf_heto_val         | 0.6199211663722822      |
| chem/raw/rank_kept            | 741                     |
| chem/white/rbf                | 0.06979717590192422     |
| chem/white/rbf_val            | 0.18793272201624245     |
| chem/white/matern52           | 0.0964006367479826      |
| chem/white/matern52_val       | 0.1883463443924033      |
| chem/white/smk                | 0.16012553216407735     |
| chem/white/smk_val            | 0.18915506873982224     |
| chem/white/rbf_heto_hetoscale | 1.0000000000000004e-06  |
| chem/white/rbf_heto           | 0.15079687131034616     |
| chem/white/rbf_heto_val       | 0.15244659731123955     |
| chem/white/rank_kept          | 741                     |

## `tdcommons/caco2-wang`  (n_tr=728, mean_absolute_error)

|                               | 0                     |
|:------------------------------|:----------------------|
| bench                         | tdcommons/caco2-wang  |
| metric                        | mean_absolute_error   |
| n_train                       | 728                   |
| n_test                        | 182                   |
| chem/raw/rbf                  | 0.3074584111786549    |
| chem/raw/rbf_val              | 0.3894291994892103    |
| chem/raw/matern52             | 0.31776888939953657   |
| chem/raw/matern52_val         | 0.39908704730939554   |
| chem/raw/smk                  | 0.2942017824957671    |
| chem/raw/smk_val              | 0.3883234804719941    |
| chem/raw/rbf_heto_hetoscale   | 4.003542368772454e-06 |
| chem/raw/rbf_heto             | 0.3192110873286754    |
| chem/raw/rbf_heto_val         | 0.4005413104047695    |
| chem/raw/rank_kept            | 538                   |
| chem/white/rbf                | 0.3824698794153227    |
| chem/white/rbf_val            | 0.6062851242022478    |
| chem/white/matern52           | 0.3901942770729573    |
| chem/white/matern52_val       | 0.623280916323497     |
| chem/white/smk                | 0.35885387277253794   |
| chem/white/smk_val            | 0.60326602562115      |
| chem/white/rbf_heto_hetoscale | 0.01088445496529202   |
| chem/white/rbf_heto           | 0.3778383629798669    |
| chem/white/rbf_heto_val       | 0.6120068620141167    |
| chem/white/rank_kept          | 538                   |
| desc/rbf                      | 0.320382454598599     |
| desc/rbf_val                  | 0.46989738116236623   |
| desc/matern52                 | 0.34444172899476094   |
| desc/matern52_val             | 0.4675060011558244    |
| desc/smk                      | 0.3480537893952123    |
| desc/smk_val                  | 0.47309858541411137   |
| desc/tanimoto_fp              | 0.329606986295878     |
| desc/tanimoto_fp_val          | 0.4659133227976041    |
| desc/tanimoto_x_rbf           | 0.3830357406907226    |
| desc/tanimoto_x_rbf_val       | 0.5022571066453883    |
| desc/rbf_heto_hetoscale       | 6.252296305154055e-06 |
| desc/rbf_heto                 | 0.314319649493258     |
| desc/rbf_heto_val             | 0.4674435293357639    |

## `tdcommons/ld50-zhu`  (n_tr=5907, mean_absolute_error)

|                               | 0                      |
|:------------------------------|:-----------------------|
| bench                         | tdcommons/ld50-zhu     |
| metric                        | mean_absolute_error    |
| n_train                       | 5907                   |
| n_test                        | 1478                   |
| chem/raw/rbf                  | 0.6241612624689441     |
| chem/raw/rbf_val              | 0.5222382102319282     |
| chem/raw/matern52             | 0.6082229154170183     |
| chem/raw/matern52_val         | 0.5205116343906571     |
| chem/raw/smk                  | 0.6135948770101397     |
| chem/raw/smk_val              | 0.5318778325818164     |
| chem/raw/rbf_heto_hetoscale   | 2.3518639909354707e-06 |
| chem/raw/rbf_heto             | 0.6248299512436843     |
| chem/raw/rbf_heto_val         | 0.5289479073402644     |
| chem/raw/rank_kept            | 1533                   |
| chem/white/rbf                | 0.6671046458188069     |
| chem/white/rbf_val            | 0.6868281804122063     |
| chem/white/matern52           | 26.778424798632486     |
| chem/white/matern52_val       | 0.6832718249621738     |
| chem/white/smk                | 0.6724629486605417     |
| chem/white/smk_val            | 0.6821993558116597     |
| chem/white/rbf_heto_hetoscale | 1.0000000000000023e-06 |
| chem/white/rbf_heto           | 0.662968590836456      |
| chem/white/rbf_heto_val       | 0.6921178919502606     |
| chem/white/rank_kept          | 1533                   |
| desc/rbf                      | 0.6781957647680585     |
| desc/rbf_val                  | 0.5950112327565776     |
| desc/matern52                 | 0.6535228810593467     |
| desc/matern52_val             | 0.5928600802996133     |
| desc/smk                      | 0.7455118848199904     |
| desc/smk_val                  | 0.6521229152683705     |
| desc/tanimoto_fp              | 0.6415790841074078     |
| desc/tanimoto_fp_val          | 0.5810418167492394     |
| desc/tanimoto_x_rbf           | 0.6516931893664203     |
| desc/tanimoto_x_rbf_val       | 0.608775788534691      |
| desc/rbf_heto_hetoscale       | 0.00024515842708824816 |
| desc/rbf_heto                 | 0.6992607593637545     |
| desc/rbf_heto_val             | 0.5886721122049438     |

# Summary (per-benchmark score; direction depends on metric)

| benchmark | chem/raw/matern52 | chem/raw/rbf | chem/raw/rbf_heto | chem/raw/smk | chem/white/matern52 | chem/white/rbf | chem/white/rbf_heto | chem/white/smk | desc/matern52 | desc/rbf | desc/rbf_heto | desc/smk | desc/tanimoto_fp | desc/tanimoto_x_rbf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polaris/pkis2-ret-wt-reg-v2 | 846.553 | 810.104 | 806.944 | 810.940 | 896.819 | 960.646 | 935.112 | 896.842 | 874.934 | 853.822 | 869.055 | 852.597 | 850.709 | 886.970 |
| polaris/pkis2-kit-wt-reg-v2 | 934.638 | 939.710 | 942.128 | 991.864 | 1090.256 | 1093.730 | 1088.580 | 1095.419 | 963.896 | 956.655 | 984.794 | 1051.556 | 937.811 | 955.839 |
| polaris/pkis2-egfr-wt-reg-v2 | 524.586 | 516.203 | 534.599 | 521.534 | 559.917 | 564.393 | 552.860 | 564.833 | 545.991 | 542.850 | 565.299 | 602.470 | 549.980 | 564.935 |
| polaris/adme-fang-solu-1 | 0.638 | 0.631 | 0.628 | 0.641 | 0.584 | 0.575 | 0.531 | 0.571 | 0.588 | 0.568 | 0.518 | 0.577 | 0.586 | 0.585 |
| polaris/adme-fang-rppb-1 | 0.538 | 0.543 | 0.508 | 0.538 | 0.796 | 0.788 | 0.795 | 0.001 | -0.122 | -0.124 | -0.125 | -0.092 | -0.121 | -0.075 |
| polaris/adme-fang-hppb-1 | 0.644 | 0.653 | 0.699 | 0.687 | 0.281 | 0.095 | 0.096 | 0.783 | 0.335 | 0.301 | 0.301 | 0.369 | 0.363 | 0.334 |
| polaris/adme-fang-perm-1 | 0.752 | 0.759 | 0.750 | 0.768 | 0.715 | 0.707 | 0.691 | 0.717 | 0.760 | 0.753 | 0.758 | 0.774 | 0.768 | 0.720 |
| polaris/adme-fang-rclint-1 | 0.716 | 0.712 | 0.701 | 0.722 | 0.616 | 0.638 | 0.618 | 0.636 | 0.667 | 0.652 | 0.635 | 0.659 | 0.671 | 0.591 |
| polaris/adme-fang-hclint-1 | 0.685 | 0.677 | 0.675 | 0.688 | 0.590 | 0.604 | 0.606 | 0.591 | 0.581 | 0.604 | 0.592 | 0.634 | 0.634 | 0.594 |
| tdcommons/lipophilicity-astrazeneca | 0.806 | 0.516 | 0.511 | 0.510 | 1.326 | 0.566 | 0.566 | 0.575 | 0.605 | 0.598 | 0.606 | 0.597 | 0.610 | 0.681 |
| tdcommons/ppbr-az | 8.985 | 8.559 | 8.236 | 8.974 | 9.947 | 9.750 | 10.257 | 10.440 | 8.792 | 8.824 | 9.089 | 9.689 | 8.882 | 9.723 |
| tdcommons/clearance-hepatocyte-az | 0.383 | 0.389 | 0.398 | 0.390 | 0.348 | 0.359 | 0.328 | 0.375 | 0.371 | 0.373 | 0.375 | 0.351 | 0.372 | 0.385 |
| tdcommons/half-life-obach | 0.225 | 0.131 | 0.343 | 0.223 | 0.106 | 0.149 | 0.061 | 0.063 | nan | nan | nan | nan | nan | nan |
| tdcommons/clearance-microsome-az | 0.574 | 0.578 | 0.492 | 0.579 | 0.384 | 0.457 | 0.415 | 0.445 | 0.560 | 0.581 | 0.496 | 0.547 | 0.568 | 0.545 |
| tdcommons/vdss-lombardo | 0.477 | 0.405 | 0.507 | 0.336 | 0.096 | 0.070 | 0.151 | 0.160 | nan | nan | nan | nan | nan | nan |
| tdcommons/caco2-wang | 0.318 | 0.307 | 0.319 | 0.294 | 0.390 | 0.382 | 0.378 | 0.359 | 0.344 | 0.320 | 0.314 | 0.348 | 0.330 | 0.383 |
| tdcommons/ld50-zhu | 0.608 | 0.624 | 0.625 | 0.614 | 26.778 | 0.667 | 0.663 | 0.672 | 0.654 | 0.678 | 0.699 | 0.746 | 0.642 | 0.652 |

# Normalized average (higher = better across all metrics)

|                     |   normalized_avg |
|:--------------------|-----------------:|
| chem/raw/rbf        |         -133.561 |
| chem/raw/rbf_heto   |         -134.568 |
| chem/raw/matern52   |         -135.933 |
| chem/raw/smk        |         -137.009 |
| chem/white/smk      |         -150.870 |
| chem/white/matern52 |         -151.819 |
| chem/white/rbf_heto |         -152.007 |
| chem/white/rbf      |         -154.452 |
| desc/tanimoto_fp    |         -156.341 |
| desc/rbf            |         -157.336 |
| desc/matern52       |         -159.432 |
| desc/tanimoto_x_rbf |         -161.034 |
| desc/rbf_heto       |         -161.754 |
| desc/smk            |         -167.612 |