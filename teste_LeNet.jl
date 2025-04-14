include("juJuBra.jl") #Biblioteca JuTorch.jl
using DelimitedFiles
using Plots

# ############################################
# # Leitura de dados
# ############################################

dados=readdlm("datasets/mnist_train.csv",',');
qnt = 60000;
X=Float32.(dados[2:qnt+1,2:end]);
L=dados[2:qnt+1,1];

if minimum(L) == 0
	L = L .+ 1;
end

L= Int64.(L); 

dados=readdlm("datasets/mnist_test.csv",',');
X2=Float32.(dados[2:end,2:end]);
L2=dados[2:end,1];

if minimum(L2) == 0
	L2 = L2 .+ 1;
end

L2= Int64.(L2); 

m=(sum(X,dims=1))/size(X,1);

X2 = (X2-ones(size(X2,1),1)*m);
X=(X-ones(size(X,1),1)*m);

X2=(1/maximum(abs.(X)))*X2;
X=(1/maximum(abs.(X)))*X;
X = [X; X2];
L = [L; L2];

# Gradiente estocástico simples com Adam
alpha = 0.01;
t = 0;
beta1 = 0.95;#0.9;
beta2 = 0.98;#0.999;
AdamEpsilon = 1e-8;

# Dimensões das camadas:
Nclasses = length(unique(L));


############################################
# Criando a topologia da Rede Neural
############################################

# Primeira Camada (convolutiva)
m1,n1,k1,c1=5,5,1,6; # dimensões dos filtros:
R1 = redeConvolutiva(1/sqrt(m1*n1*k1)*(randn(Float32,m1,n1,k1,c1)), #W
						zeros(1,1,1,c1), #B
						x -> max(x,0), #f(x) = Relu(x)
						f -> Float32(f > 0), #f'(x)
						1, #Stride
						[28,28] #Dimensões da entrada
						);
dc1 = prod(R1.Out);

# Segunda camada (max pooling):
R2 = redeMaxPooling(R1.Out, #dimensões da entrada
                    2, #Stride
                    [2,2], #Dimensões do Pooling
                    );
dc2 = prod(R2.Out);

# Terceira camada (Convolucional):
m2,n2,k2,c2=5,5,6,16; # dimensões dos filtros:
R3 = redeConvolutiva(1/sqrt(m2*n2*k2)*(randn(Float32,m2,n2,k2,c2)), #W
						zeros(1,1,1,c2), #B
						x -> max(x,0), #f(x) = Relu(x)
						f -> Float32(f > 0), #f'(x)
						1, #Stride
						R2.Out #Dimensões da entrada
						);
dc3 = prod(R3.Out);

# Quarta camada (max pooling):
R4 = redeMaxPooling(R3.Out, #dimensões da entrada
                    2, #Stride
                    [2,2], #Dimensões do Pooling
                    );
dc4 = prod(R4.Out);

# Quinta camada (densa):
dc5 = 120; # Número de neurônios na primeira camada densa

R5 = redeDensa(1/sqrt(dc5)*randn(Float32,dc5,dc4), # W
                zeros(dc5,1), # B
                x -> 1/(1 + exp(-x)), #f(x)
                f -> f * (1-f)); #df(x)

# Sexta camada (densa):
dc6 = 84; # Número de neurônios na primeira camada densa

R6 = redeDensa(1/sqrt(dc6)*randn(Float32,dc6,dc5), # W
                zeros(dc6,1), # B
                x -> 1/(1 + exp(-x)), #f(x)
                f -> f * (1-f)); #df(x)

# Quarta camada (densa):
dc7 = Nclasses;
R7 = redeDensa(1/sqrt(dc7)*randn(Float32,dc7,dc6), # W
                zeros(dc7,1), # B
                x -> 1/(1 + exp(-x)), #f(x)
                f -> f * (1-f)); #df(x)

# Ida
function redeIda(n)
    X0 .= Array(reshape(X[n,:],R1.In...)) #X0: entrada da rede');

	# Camada convolutiva (1):
	redeGenericaIda(R1,X0,X1)

    # Camada convolutiva (1):
    redeGenericaIda(R2,X1,X2)

    # Camada densa (2):
	redeGenericaIda(R3,X2,X3);

	# Camada densa (3):
	redeGenericaIda(R4,X3,X4);

	# Camada densa (3):
	redeGenericaIda(R5,X4,X5);

	# Camada densa (3):
	redeGenericaIda(R6,X5,X6);

	redeGenericaIda(R7,X6,X7);

    # Cálculo do erro:
    O .= 0;
    O[convert(Int32,L[n])]=1;

    E7 .= (X7 .- O);
end

# Volta
function redeVolta()
	global t
	t += 1;
	redeGenericaVolta(R7,X6,X7,E6,E7);
	redeGenericaVolta(R6,X5,X6,E5,E6);
	redeGenericaVolta(R5,X4,X5,E4,E5);
    redeGenericaVolta(R4,X3,X4,E3,E4);
    redeGenericaVolta(R3,X2,X3,E2,E3);
	redeGenericaVolta(R2,X1,X2,E1,E2)
	redeGenericaVolta(R1,X0,X1,E0,E1)
end

#Inicialização da variáveis de treinamento:
X0 = zeros(Float32,R1.In...);
X1 = zeros(Float32,R1.Out...);
X2 = zeros(Float32,R2.Out...);
X3 = zeros(Float32,R3.Out...);
X4 = zeros(Float32,R4.Out...);
X5 = zeros(Float32,R5.Out);
X6 = zeros(Float32,R6.Out);
X7 = zeros(Float32,R7.Out);

E0 = zeros(Float32,R1.In...);
E1 = zeros(Float32,R1.Out...);
E2 = zeros(Float32,R2.Out...);
E3 = zeros(Float32,R3.Out...);
E4 = zeros(Float32,R4.Out...);
E5 = zeros(Float32,R5.Out);
E6 = zeros(Float32,R6.Out);
E7 = zeros(Float32,R7.Out);
O  = zeros(Float32,R7.Out)

# Sorteio:
aux=rand(size(X,1));
#tudo=sortperm(aux);
tudo = 1:size(X,1);
#pLimiar = round(Int32,size(X,1)*0.7);
pLimiar = qnt;
pTreino = tudo[1:pLimiar];
pTeste = tudo[pLimiar+1:end];

############################################
# Medidas de desempenho
############################################
Nciclos = 50;
J=zeros(Float32,Nciclos,1);
A = zeros(Float32,Nciclos,1);
Jteste=zeros(Float32,Nciclos,1);
Ateste = zeros(Float32,Nciclos,1)

for ciclo in 1:Nciclos
	for n in pTeste

		redeIda(n);

		# Medidas de desempenho:
		Jteste[ciclo] += sum(E7.^2)/length(E7);
		if (argmax(O)==argmax(X7))
			Ateste[ciclo] += 1;
		end

	end
	
	for n in pTreino
	
		redeIda(n);

		# Medidas de desempenho:
		J[ciclo] += sum(E7.^2)/length(E7);
		if (argmax(O)==argmax(X7))
			A[ciclo] += 1;
		end
		
		redeVolta();
	end


	J[ciclo] = J[ciclo]/length(pTreino);
	A[ciclo] = A[ciclo]/length(pTreino);
	
	Jteste[ciclo] = Jteste[ciclo]/length(pTeste);
	Ateste[ciclo] = Ateste[ciclo]/length(pTeste);
	
	pJ=plot(J, label = "EQM train");
	pJt=plot!(Jteste, label = "EQM test");
	
	pA=plot(A,label ="Acuraccy train");
	pAt=plot!(Ateste, label = "Acuraccy test");
	
	pz=plot(pJt,pAt, layout=grid(2,1))
	
	display(pz)
	
	println("Ciclo: ", ciclo, " Taxa de acerto (treino): ", 100*round(A[ciclo],digits =4), "%", "\n");
	println("Ciclo: ", ciclo, " Taxa de acerto (teste): ", 100*round(Ateste[ciclo],digits =4), "%", "\n");

end