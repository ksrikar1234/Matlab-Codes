Design and simulate the characteristics of first and second order circuits in time and frequency domain

clc;
clear;
 R=1; 
L=0.5;
G=tf ([0 1],[R L])
dobe(G)

R=1
C=0.5
G=tf ([0 1],[1 R*C])
 dobe(G)

R=1
L=0.5
C=0.5
G=tf ([0 1],[1 R/L 1/L*C])
 dobe(G)

------------------------------------------------------------------------------------------------------------------------------------------------------------
Design and analysis of the performance of feedback control system

s = tf('s');
G = 20/(s^2+3*s+20);
k = 0.7;
A = k*G;
figure(1);
pzmap(A);
grid on;
figure(2);
step(A);
grid on;
A1 = feedback(A,1,-1);
figure(3);
pzmap(A1);
grid on;
figure(4);
step(A1);
grid on;
A2 = feedback(A,1,1);
figure(5);
pzmap(A2);
grid on;
figure(6);
step(A2);
grid on;
---------------------------------------------------------------------------------------------------------------------------------------------------

TRANSMISSION LINES

NOMINAL T
clc;
clear all;
disp('***Enter the system data*')
x=input('Length of transmission line in km: ');
Vrl=input('\n Receiving end voltage in kv: ');
r=input('\n Per km resistance in ohm: ');
l=input('\n Per km inductance/phase in mH: ');
c=input('\n Per km capacitance/phase in uF: ');
P=input('\n Receiving power in MW: ');
pf=input('\n Lagging power factor: ');
f =input('\n frequency: ');
S=(P/pf)*(10^6);
Vrl=Vrl*(10^3);
l=l*(10^-3);
c=c*(10^-6);
L=x*l;
R=r*x;
C=c*x;
pf_angle = acos(pf);
Vr=Vrl/sqrt(3); %Receiving end phase voltage
Z=complex(R,2*pi*50*L); 
Vr = complex(Vr,0); %Receiving end phase voltage as reference Vr at angle 0;
j=sqrt(-1);
Y = j*2*pi*f*c*x;
Ir=S/(3*Vr); %Receiving end current
Ir=Ir*complex(pf,-sin(pf_angle)); %Receiving end current in complex form
%% calculating ABCD parameters
A= 1+(Z*Y/2);
B= Z*(1+Z*Y/4);
C=Y;
D=1+(Z*Y/2);
%% calculation of sending end voltage and receiving end voltage from branch currents
Vc=Vr+Ir*(Z/2);
Ic=Vc*Y;
Isb=Ir+Ic; %sending end current 
Vsb = Vc+Isb*Z/2 ; %Sending end phase voltage
%% calculation of sending voltage and current
Vs = A*Vr+B*Ir; %Sending end phase voltage
Is = C*Vr+D*Ir; %sending end current
%% Regulation
Vr_NL = Vs*(1/Y)/(Z/2+(1/Y));
regulation = 100*abs((Vr_NL-Vr)/Vr_NL); %Voltage Regulation
regulation_ABCD = 100*abs(((Vs/A)-Vr)/(Vr)); %Voltage Regulation using ABCD parameters
%% Efficiency
Ploss = 3*abs(Ir^2)*R/(10^6);
Efficiency = 100*(P/(P+Ploss));
%% Output the transmissionline parameters
disp('The Sending end Voltages and currents:');
fprintf('\n per phase Vs = %f \t Is = %f',Vs,Is);
fprintf('\n');
disp('The ABCD parameters are:');
fprintf('\nA = %f \t B = %f',A,B);
fprintf('\nC = %f \t D = %f \n',C,D);
fprintf('\n');
 
disp('The Regulation and Efficiency are:');
fprintf('\n Regulation = %f \t Efficiency = %f \n ',regulation,Efficiency);

------------------------------------------------------------------------------------------------------------------------------
            PI model
clc;
clear all;
disp('******Enter the system data****')
x=input('Length of transmission line in km: ');
Vrl=input('\n Receiving end voltage in kv: ');
r=input('\n Per km resistance in ohm: ');
l=input('\n Per km inductance/phase in mH: ');
c=input('\n Per km capacitance/phase in uF: ');
P=input('\n Receiving power in MW: ');
pf=input('\n Lagging power factor: ');
f =input('\n frequency: ');
S=(P/pf)*(10^6);
Vrl=Vrl*(10^3);
l=l*(10^-3);
c=c*(10^-6);
L=x*l;
R=r*x;
Cs=c*x;
pf_angle = acos(pf);
Vr=Vrl/sqrt(3); 
Z=complex(R,2*pi*50*L); 
Vr = complex(Vr,0); 
j=sqrt(-1);
Ir=S/(3*Vr); 
Ir=Ir*complex(pf,-sin(pf_angle));
Y = j*2*pi*f*Cs; 
Ic1=Vr*Y/2;
Il=Ir +Ic1;
A= 1+(Z*Y/2);
B= Z;
C=Y*(1+Z*Y/4);
D=1+(Z*Y/2); 
Vsb = Vr+Il*Z; 
Ic2=Vsb*Y/2;
Isb = Ic2+Il; 
Vs = A*Vr+B*Ir; 
Is = C*Vr+D*Ir;
Vr_NL = Vs*(-2/Y)/(Z-2/Y);
regulation = 100*abs((Vr_NL-Vr)/Vr_NL); 
regulation_ABCD = 100*abs(((Vs/A)-Vr)/(Vs/A));
Ploss = 3*abs(Ir^2)*R/(10^6);
Efficiency = 100*(P/(P+Ploss));
disp('The Sending end Voltages and currents of Nominal -Pi:');
fprintf('\n per phase Vs = %f \t Is = %f',Vs,Is);
fprintf('\n');
disp('The ABCD parameters are:');
fprintf('\nA = %f \t B = %f',A,B);
fprintf('\nC = %f \t D = %f \n',C,D);
disp('The Regulation and Efficiency are:');
fprintf('\n Regulation = %f \t Efficiency = %f \n ',regulation,Efficiency);
------------------------------------------------------------------------------------------------------

long tr
clc;
clear all;
x = input('Length of transmission line in km: ');
f = input('frequency of transmission line in Hz: ');
sr = input('Receiving end power in MVA: ');
pf = input('Lagging power factor: ');
Vrl = input('Receiving end voltage in KV: ');
r = input('Resistance/phase in ohm/km: ');
l = input('Inductance/phase in mH/km: ');
c = input('Capacitance/phase in uF/km: ');
R = r*x;
L = (l*x)*10^-3;
C = (c*x)*10^-6;
Z = complex(R,2*pi*f*L);
Y = complex(0,2*pi*f*C);
Zc = sqrt(Z/Y);
vl = sqrt(Z*Y);
%calculating ABCD parameters;
A = cosh(vl);
B = Zc*sinh(vl);
C = (1/Zc)*sinh(vl);
D = cosh(vl);
%Calculating receiving end current;
Sr = sr*10^6;
Vr = (Vrl/sqrt(3))*10^3;
ir = Sr/(3*Vr);
pf_angle = acos(pf);
Ir = ir*complex(pf,-sin(pf_angle));
%calculating sending end voltage and current;
Vs = A*Vr+B*Ir;
Is = C*Vr+D*Ir;
%Calculating Voltage regulation and Efficiency;
VR = 100*abs((Vs/A)-Vr)/Vr;
ploss = 3*abs(Is^2)*R;
Pr = Sr*pf;
Ef = 100*Pr/(Pr+ploss);
%Display the data;
fprintf('\n');
disp('The ABCD parameters are: ');
fprintf('\n A = %f \t B = %f ',A,B);
fprintf('\n C = %f \t D = %f ',C,D);
fprintf('\n');
disp('Voltage regulation and efficiency are: ');
fprintf('\n Voltage regulation = %f \t Efficiency = %f ',VR,Ef);
----------------------------------------------------------------------------------------

pid

s = tf('s');
G= 1/(s^2 + 10*s +20);
figure(1)
step(G)
Kp=300;
C=pid(Kp)
plant=C*G;
T1=feedback(plant,1)
t=0:0.01:2;
figure(2)
step(T1,t)
Kp=300;
Kd=10;
C=pid(Kp,0,Kd)
plant=C*G;
T2=feedback(plant,1)
t=0:0.01:2;
figure(3)
step(T2,t)
Kp=300;
Ki=70;
C=pid(Kp,Ki)
plant=C*G;
T3=feedback(plant,1)
t=0:0.01:2;
figure(4)
step(T3,t)
Kp=350;
Ki=300;
Kd=50;
C=pid(Kp,Ki,Kd)
plant=C*G;
T4=feedback(plant,1)
t=0:0.01:2;
figure(5)
step(T4,t)
Kp=350;
Ki=300;
Kd=50;
C=pid(Kp,Ki,Kd)
plant=C*G;
T5=feedback(plant,1)
t=0:0.01:2;
figure(5)
step(T5,t)
---------------------------------------------------------------------------------------------------------
symmetrical compound analysis

clc;
clear;
disp('Provide the abc phase componenets');
Va_mag = input('a phase magnitude: ');
Va_ang = input('a phase angle: ');
Vb_mag = input('b phase magnitude: ');
Vb_ang = input('b phase angle: ');
Vc_mag = input('c phase magnitude: ');
Vc_ang = input('c phase angle: ');
Va = Va_mag*exp(j*Va_ang*(pi/180));
Vb = Vb_mag*exp(j*Vb_ang*(pi/180));
Vc = Vc_mag*exp(j*Vc_ang*(pi/180));
%Sequence transformation Matrix
alpha = 1*exp(j*2*pi/3);
V_abc = [Va;Vb;Vc];
%sequence transformation from abc phase to 012 sequence
ph_sym = (1/3)*[1 1 1;1 alpha alpha^2;1 alpha^2 alpha]*V_abc
%Zero sequence of abc phase
Va0_mag = abs(ph_sym(1));
Va0_ang = angle(ph_sym(1))*180/pi;
 
Vb0_mag = Va0_mag;
Vb0_ang = Va0_ang;
 
Vc0_mag = Va0_mag;
Vc0_ang = Va0_ang;
%Positive sequence
Va1_mag = abs(ph_sym(2));
Va1_ang = angle(ph_sym(2))*180/pi;
 
[r,i] = pol2cart(Va1_ang*pi/180,Va1_mag);
 
Vb1 = (alpha^2)*complex(r,i);
Vb1_mag = abs(Vb1);
Vb1_ang = angle(Vb1)*180/pi;
 
Vc1 = (alpha)*complex(r,i);
Vc1_mag = abs(Vc1);
Vc1_ang = angle(Vc1)*180/pi;
%Negative sequence
Va2_mag = abs(ph_sym(3));
Va2_ang = angle(ph_sym(3))*180/pi;
 
[r,i] = pol2cart(Va2_ang*pi/180,Va2_mag);
 
Vb2 = (alpha)*complex(r,i);
Vb2_mag = abs(Vb2);
Vb2_ang = angle(Vb2)*180/pi;
 
Vc2 = (alpha^2)*complex(r,i);
Vc2_mag = abs(Vc2);
Vc2_ang = angle(Vc2)*180/pi;
disp('The symmetrical components of abc phase are ');
disp('Sequence Magnitude Angle');
disp('_________________________ ');
fprintf('Va0\t%f\t%f',Va0_mag,Va0_ang);
fprintf('\nVa1\t%f\t%f',Va1_mag,Va1_ang);
fprintf('\nVa2\t%f\t%f\n',Va2_mag,Va2_ang);
fprintf('\nVb0\t%f\t%f',Vb0_mag,Vb0_ang);
fprintf('\nVb1\t%f\t%f',Vb1_mag,Vb1_ang);
fprintf('\nVb2\t%f\t%f\n',Vb2_mag,Vb2_ang);
fprintf('\nVc0\t%f\t%f',Vc0_mag,Vc0_ang);
fprintf('\nVc1\t%f\t%f',Vc1_mag,Vc1_ang);
fprintf('\nVc2\t%f\t%f',Vc2_mag,Vc2_ang);
 
%Transformation from sequence componenets to abc phase
seq_012 = ph_sym;
sym_ph = [1 1 1;1 alpha^2 alpha;1 alpha alpha^2]*ph_sym;
%a phase components
VA_mag = abs(sym_ph(1));
VA_ang = angle(sym_ph(1))*180/pi;
%b phase components
VB_mag = abs(sym_ph(2));
VB_ang = angle(sym_ph(2))*180/pi;
%c phase components
VC_mag = abs(sym_ph(3));
VC_ang = angle(sym_ph(3))*180/pi;
 
fprintf('\n\n');
disp('The phase components from sequence components');
disp('phase Magnitude Angle');
disp('_____________________');
fprintf('Va\t%f\t%f',VA_mag,VA_ang);
fprintf('\nVb\t%f\t%f',VB_mag,VB_ang);
fprintf('\nVc\t%f\t%f',VC_mag,VC_ang);







