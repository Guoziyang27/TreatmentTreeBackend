select stay_id as icustay_id, UNIX_SECONDS(TIMESTAMP(charttime)) as charttime, itemid, valuenum
from `physionet-data.mimic_icu.chartevents`
where valuenum is not null and stay_id is not null and itemid in  (829,	1535,	227442,	227464,	4195	,3726	,3792,837,	220645,	4194,	3725,	3803	,226534,	1536,	4195,	3726,788,	220602,	1523,	4193,	3724	,226536,	3747,225664,	807,	811,	1529,	220621,	226537,	3744,781,	1162,	225624,	3737,791,	1525,	220615,	3750,821,	1532,	220635,786,	225625,	1522,	3746,816,	225667,	3766,777,	787,770,	3801,769,	3802,1538,	848,	225690,	803,	1527,	225651,	3807,	1539,	849,	772,	1521,	227456,	3727,	227429,	851,227444,	814,	220228,	813,	220545,	3761,	226540,	4197,	3799	,1127,	1542,	220546,	4200,	3834,	828,	227457,	3789,825,	1533,	227466,	3796,824,	1286,1671,	1520,	768,220507	,815,	1530,	227467,	780,	1126,	3839,	4753,779,	490,	3785,	3838,	3837,778,	3784,	3836,	3835,776,	224828,	3736,	4196,	3740,	74,225668,1531,227443,1817,	228640,823,	227686, 220587, 227465, 220224, 226063, 226770, 227039, 220235, 226062, 227036)
order by icustay_id, charttime, itemid