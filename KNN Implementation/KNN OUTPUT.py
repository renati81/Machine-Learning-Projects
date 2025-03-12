Python 3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

= RESTART: C:\Users\HP\OneDrive\Desktop\UTA Docs\Spring 2025\CSE 6363 - Machine Learning\Assignments\KNN\KNN Code.py

Processing Data Source: hayes-roth

Loaded hayes-roth data. Dimensions: (133, 6)
      0      1    2                3               4      5
0  name  hobby  age  education level  marital status  class
1    92      2    1                1               2      1
2    10      2    1                3               2      2
3    83      3    1                4               1      3
4    61      2    4                2               2      3

Identifying missing values before processing the data:
No missing values found. 
Skipping missing value handling.

Data preview:
     0  1  2  3  4  5
0  132  3  4  4  4  3
1  124  1  0  0  1  0
2    1  1  0  2  1  1
3  114  2  0  3  0  2
4   90  1  3  1  1  2
Features prepared. Statistics:
Average: -0.0000, Standard Deviation: 1.0000

K=1, Validation Accuracy: 0.3333

K=3, Validation Accuracy: 0.1852

K=5, Validation Accuracy: 0.2222

K=7, Validation Accuracy: 0.2593

K=9, Validation Accuracy: 0.3333
Optimal K: 1 with accuracy: 0.3333
K=1
My KNN (Avg Accuracy): 0.6319
Sklearn KNN (Avg Accuracy): 0.6319
K=3
My KNN (Avg Accuracy): 0.5940
Sklearn KNN (Avg Accuracy): 0.5940
K=5
My KNN (Avg Accuracy): 0.5802
Sklearn KNN (Avg Accuracy): 0.5802
K=7
My KNN (Avg Accuracy): 0.5802
Sklearn KNN (Avg Accuracy): 0.5802
K=9
My KNN (Avg Accuracy): 0.5725
Sklearn KNN (Avg Accuracy): 0.5725

Paired T-Test for K=1
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=3
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=5
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=7
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=9
T-statistic = nan
P-value = nan
No Significant Difference Found.
Final Test Accuracy for hayes-roth with K=1: 0.6296

Processing Data Source: car

Loaded car data. Dimensions: (1729, 7)
         0      1      2        3         4       5      6
0  buying   maint  doors  persons  lug_boot  safety  class
1    vhigh  vhigh      2        2     small     low  unacc
2    vhigh  vhigh      2        2     small     med  unacc
3    vhigh  vhigh      2        2     small    high  unacc
4    vhigh  vhigh      2        2       med     low  unacc

Identifying missing values before processing the data:
No missing values found. 
Skipping missing value handling.

Data preview:
   0  1  2  3  4  5  6
0  0  2  4  3  1  3  1
1  4  4  0  0  3  1  3
2  4  4  0  0  3  2  3
3  4  4  0  0  3  0  3
4  4  4  0  0  2  1  3
Features prepared. Statistics:
Average: -0.0000, Standard Deviation: 1.0000

K=1, Validation Accuracy: 0.7861

K=3, Validation Accuracy: 0.8497

K=5, Validation Accuracy: 0.8410

K=7, Validation Accuracy: 0.8237

K=9, Validation Accuracy: 0.8121
Optimal K: 3 with accuracy: 0.8497
K=1
My KNN (Avg Accuracy): 0.7247
Sklearn KNN (Avg Accuracy): 0.7247
K=3
My KNN (Avg Accuracy): 0.8485
Sklearn KNN (Avg Accuracy): 0.8502
K=5
My KNN (Avg Accuracy): 0.8230
Sklearn KNN (Avg Accuracy): 0.8155
K=7
My KNN (Avg Accuracy): 0.7912
Sklearn KNN (Avg Accuracy): 0.7900
K=9
My KNN (Avg Accuracy): 0.8103
Sklearn KNN (Avg Accuracy): 0.8080

Paired T-Test for K=1
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=3
T-statistic = -1.9640
P-value = 0.0811
No Significant Difference Found.

Paired T-Test for K=5
T-statistic = 4.3362
P-value = 0.0019
Significant Difference Found!

Paired T-Test for K=7
T-statistic = 1.5000
P-value = 0.1679
No Significant Difference Found.

Paired T-Test for K=9
T-statistic = 1.3107
P-value = 0.2224
No Significant Difference Found.
Final Test Accuracy for car with K=3: 0.8035

Processing Data Source: breast-cancer

Loaded breast-cancer data. Dimensions: (287, 10)
                      0      1          2  ...       7            8         9
0                 Class    age  menopause  ...  breast  breast-quad  irradiat
1  no-recurrence-events  30-39    premeno  ...    left     left_low        no
2  no-recurrence-events  40-49    premeno  ...   right     right_up        no
3  no-recurrence-events  40-49    premeno  ...    left     left_low        no
4  no-recurrence-events  60-69       ge40  ...   right      left_up        no

[5 rows x 10 columns]

Identifying missing values before processing the data:
No missing values found. 
Skipping missing value handling.

Data preview:
   0  1  2   3  4  5  6  7  8  9
0  0  6  2  11  7  2  3  0  1  0
1  1  1  3   5  0  1  2  1  3  1
2  1  2  3   3  0  1  1  2  6  1
3  1  2  3   3  0  1  1  1  3  1
4  1  4  0   2  0  1  1  2  4  1
Features prepared. Statistics:
Average: -0.0000, Standard Deviation: 1.0000

K=1, Validation Accuracy: 0.6491

K=3, Validation Accuracy: 0.7018

K=5, Validation Accuracy: 0.7368

K=7, Validation Accuracy: 0.7544

K=9, Validation Accuracy: 0.7895
Optimal K: 9 with accuracy: 0.7895
K=1
My KNN (Avg Accuracy): 0.7039
Sklearn KNN (Avg Accuracy): 0.7038
K=3
My KNN (Avg Accuracy): 0.7596
Sklearn KNN (Avg Accuracy): 0.7596
K=5
My KNN (Avg Accuracy): 0.7772
Sklearn KNN (Avg Accuracy): 0.7738
K=7
My KNN (Avg Accuracy): 0.7739
Sklearn KNN (Avg Accuracy): 0.7739
K=9
My KNN (Avg Accuracy): 0.7633
Sklearn KNN (Avg Accuracy): 0.7633

Paired T-Test for K=1
T-statistic = 0.0168
P-value = 0.9870
No Significant Difference Found.

Paired T-Test for K=3
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=5
T-statistic = 1.0000
P-value = 0.3434
No Significant Difference Found.

Paired T-Test for K=7
T-statistic = nan
P-value = nan
No Significant Difference Found.

Paired T-Test for K=9
T-statistic = nan
P-value = nan
No Significant Difference Found.
Final Test Accuracy for breast-cancer with K=9: 0.6724
