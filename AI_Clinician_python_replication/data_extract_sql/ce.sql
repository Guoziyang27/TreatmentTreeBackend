select distinct stay_id as icustay_id, UNIX_SECONDS(TIMESTAMP(charttime)) as charttime, itemid, 
case 
when value = 'None' then 0 
when value = 'Ventilator' then 1 
when value='Cannula' then 2 
when value = 'Nasal Cannula' then 2 
when value = 'Face Tent' then 3 
when value = 'Aerosol-Cool' then 4 
when value = 'Trach Mask' then 5 
when value = 'Hi Flow Neb' then 6 
when value = 'Non-Rebreather' then 7 
when value = '' then 8  
when value = 'Venti Mask' then 9 
when value = 'Medium Conc Mask' then 10 
else valuenum end as valuenum 
from `physionet-data.mimic_icu.chartevents` 
where stay_id>={} and stay_id<{} and value is not null and itemid in (226707, 581, 198, 228096, 211, 220179, 220181, 8368, 220210, 220277, 3655, 223761, 220074, 492, 491, 8448, 116, 626, 467, 223835, 190, 470, 220339, 224686, 224687, 224697, 224695, 224696, 226730, 580, 220045, 225309, 220052, 8441, 3337, 646, 223762, 678, 113, 1372, 3420, 471, 506, 224684, 450, 444, 535, 543, 224639, 6701, 225312, 225310, 224422, 834, 1366, 160, 223834, 505, 684, 448, 226512, 6, 224322, 8555, 618, 228368, 727, 227287, 224700, 224421, 445, 227243, 6702, 8440, 3603, 228177, 194, 3083, 224167, 443, 615, 224691, 2566, 51, 52, 654, 455, 456, 3050, 681, 2311)  
order by icustay_id, charttime
