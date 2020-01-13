#Importing Required Libraries
import matplotlib.pyplot as plt
from datetime import datetime
import operator
import pandas as pd
import numpy as np
import math
from math import pow
import csv

#Converting Cartesian inputs into magnitude
def c2m(x, y, d, c):
  top = np.sqrt((x*x) + (y*y))
  num = c * top
  total = num/d

  return total

#Converting db back to standard numbers
def db2lin(A, P1 = 1):
  x = math.pow(10,(A/10))
  P2 = P1 * x

  return P2

#Converting standard numbers to db scale 
def lin2db(P2, P1):
  A = 10*np.log(P2/P1)
  return A

#Converting Cartesian inputs into Polar coordinates
def c2p(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    #phi = math.degrees(np.arctan2(y, x))
    return rho, phi

#Converting Polar Inputs into Cartesian coordinates
def p2c(rho, phi):
    phi = np.radians(phi)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

#Opening and Reading Data Files
data = pd.read_csv(".//Input//D3_Input1.csv")

#Admitance to be y axis
y = []
eq1 = []
eq2 = []
s11_data = []
y_real = []
y_imag = []
imp_real = []
imp_imag = []
imped_data = []
eq_lin_mag = []
eq_lin_real = []
eq_lin_imag = []


#Frequency is x axis
x = data['Freq(Hz)'].tolist()

#Creating a Dataframe to Save File
dataframe = pd.DataFrame()


# Loading the data into arrays
S11db = data['S11(DB)'].tolist()
S11deg = data['S11(DEG)'].tolist()
S12db = data['S12(DB)'].tolist()
S12deg = data['S12(DEG)'].tolist()
S21db = data['S21(DB)'].tolist()
S21deg = data['S21(DEG)'].tolist()
S22db = data['S22(DB)'].tolist()
S22deg = data['S22(DEG)'].tolist()
EQdb = data['Eq=(1-S11)/(1+S11)(DB)'].tolist()
EQdeg = data['Eq=(1-S11)/(1+S11)(DEG)'].tolist()


#Calculating Admittance based on Scattering Parameter
for index,item in enumerate(S11db):

  #Math based on the information in the Google Docs

  EQ_db = EQdb[index]

  EQ_deg = EQdeg[index]
  
  eq_lin_mag1 = db2lin(EQ_db)
  
  eq_lin_real1, eq_lin_img1 = p2c(eq_lin_mag1, EQ_deg)
  
  S11_mag = db2lin(S11db[index])

  A, B = p2c(S11_mag, S11deg[index]) 
  # What is the db value in reference to? In this program it is 1.

  paper_eq1 = (1-A*A-B*B)
  paper_eq2 = (-2*B)
  paper_eqden = (A*A + 2*A + 1 + B*B)

  Z_test = (1/50)

  Y_mag = c2m(paper_eq1, paper_eq2, paper_eqden, Z_test)

  #Y_mag_logged = lin2db(Y_mag, 1) - Logging the result, uncomment if needed
 
  #Adding each admitance to list y
  y.append(Y_mag)
  eq1.append(EQ_db)
  eq2.append(EQ_deg)
  s11_data.append(S11_mag)
  imped_data.append(1/Y_mag)
  y_real.append(Z_test*paper_eq1/paper_eqden)
  y_imag.append(Z_test*paper_eq2/paper_eqden)
  imp_real.append(paper_eqden/(Z_test*paper_eq1))
  imp_imag.append(1/((Z_test*(1/paper_eqden))*(paper_eq2)))
  eq_lin_mag.append(eq_lin_mag1)
  eq_lin_real.append(eq_lin_real1)
  eq_lin_imag.append(eq_lin_img1)

#Setting Up to Write to CSV
labels = ['Frequency', 'EQ (db)', 'EQ (deg)', 'EQ mag', 'EQ Real', 'EQ Imagi', 'Calculated Admittance Magnitude', 'Calculated Admitance Real', 'Calculated Admitance Imagi', 'Calculated Impedance Magnitude', 'Calculated Impedance Real', 'Calculated Impedance Imagi', 'S11 mag linear']
rows = zip(x, eq1, eq2, eq_lin_mag, eq_lin_real, eq_lin_imag, y, y_real, y_imag, imped_data, imp_real, imp_imag, s11_data)
newfilePath = "data_converted.csv"

#Writing to a File
with open(newfilePath, "w") as f:
    writer = csv.writer(f)
    writer.writerow(labels)
    for row in rows:
        writer.writerow(row)

#Plotting the Data
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()

ax1.set_title('Admittance Magnitude vs Frequency')
ax1.plot(x, y, label='Caculated Admittance')
ax1.plot(x, eq_lin_mag, label='Measured Admitance')
ax1.legend()
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Admittance')
ax1.grid(True)
fig1.savefig("Admittance_Mag_vs_Frequency_Data"+str(datetime.now())+".png")

ax2.set_title('Admitance Real Vs Frequency')
ax2.plot(x,y_real, label='Calculated Admitance Real')
ax2.plot(x,eq_lin_real, label='Measured Admitance Real')
ax2.legend()
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Admittance')
ax2.grid(True)
fig2.savefig("Admittance_Real_vs_Frequency_Data"+str(datetime.now())+".png")

ax3.set_title('Admitance Imaginary Vs Frequency')
ax3.plot(x,y_imag, label='Calculated Admitance Imagi')
ax3.plot(x,eq_lin_imag, label='Measured Admitance Imagi')
ax3.legend()
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Admittance')
ax3.grid(True)
fig3.savefig("Admittance_Imag_vs_Frequency_Data"+str(datetime.now())+".png")

ax4.set_title('Impedance magnitude Vs Frequency')
ax4.plot(x,imped_data, label='Calculated Impedance Magnitude')
ax4.legend()
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Impedance (Ohms)')
ax4.grid(True)
fig4.savefig("Impedance_Mag_vs_Frequency_Data"+str(datetime.now())+".png")

ax5.set_title('Impedance magnitude Real Vs Frequency')
ax5.plot(x,imp_real, label='Calculated Impedance Real')
ax5.legend()
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Impedance (Ohms)')
ax5.grid(True)
fig5.savefig("Impedance_Real_vs_Frequency_Data"+str(datetime.now())+".png")

ax6.set_title('Impedance Imaginary Vs Frequency')
ax6.plot(x,imp_imag, label='Calculated Impedance Imagi')
ax6.legend()
ax6.set_xlabel('Frequency (Hz)')
ax6.set_ylabel('Impedance (Ohms)')
ax6.grid(True)
fig6.savefig("Impedance_Imag_vs_Frequency_Data"+str(datetime.now())+".png")

plt.show()

#Calculating Peak Fs - Still need to devise a better method of calculation (Save to file)
max_index, max_value = max(enumerate(y), key=operator.itemgetter(1))

min_index, min_value = min(enumerate(y), key=operator.itemgetter(1))

X_min = x[min_index]
X_max = x[max_index]

predicted_Peak_fs = ((X_min+X_max)/2)

print("Predicted Peak Fs in Admitance:" + str(predicted_Peak_fs))

#MBVD Fitting Based on sensor.png which comes from this paper

#https://www.researchgate.net/publication/321406283_Extraction_of_modified_butterworth_-_Van_Dyke_model_of_FBAR_based_on_FEM_analysis