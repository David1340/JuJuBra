include("juJuBra.jl") #Biblioteca JuTorch.jl
using DelimitedFiles
using Plots

# ############################################
# # Leitura de dados
# ############################################

dados=readdlm("datasets/mnist_3000.txt");
X=dados[:,2:end];
L=dados[:,1];

if minimum(L) == 0
	L = L .+ 1;
end

L= Int64.(L); 
m=(sum(X,dims=1))/size(X,1);
X=(X-ones(size(X,1),1)*m);
X=(1/maximum(abs.(X)))*X;


# Gradiente estocástico simples
alpha = 0.01;

# Dimensões das camadas:
Nclasses = length(unique(L));


############################################
# Criando a topologia da Rede Neural
############################################

# Primeira Camada (convolutiva)
m1,n1,k1,c1=5,5,1,10; # dimensões dos filtros:
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
                    1, #Stride
                    [2,2], #Dimensões do Pooling
                    );
dc2 = prod(R2.Out);

# Terceira camada (densa):
dc3 = 50; # Número de neurônios na primeira camada densa

R3 = redeDensa(1/sqrt(dc2)*randn(Float32,dc3,dc2), # W
                zeros(dc3,1), # B
                x -> 1/(1 + exp(-x)), #f(x)
                f -> f * (1-f)); #df(x)

# Quarta camada (densa):
dc4 = Nclasses;
R4 = redeDensa(1/sqrt(dc3)*randn(Float32,dc4,dc3), # W
                zeros(dc4,1), # B
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

    # Cálculo do erro:
    O .= 0;
    O[convert(Int32,L[n])]=1;

    E4 .= (X4 .- O);
end

# Volta
function redeVolta()

    # Retropropagação pela segunda camada densa:
    voltaDensa(R4,X3,X4,E3,E4);

    # Retropropagação pela primeira camada densa:
    voltaDensa(R3,X2,X3,E2,E3);

	# Retropropagação pela primeira camada Convolutiva:
	voltaMaxPooling(R2,X1,X2,E1,E2)
    # Retropropagação pela primeira camada Convolutiva:
	voltaConv(R1,X0,X1,E0,E1)
end

#Inicialização da variáveis de treinamento:
X0 = zeros(Float32,R1.In...);
X1 = zeros(Float32,R1.Out...);
X2 = zeros(Float32,R2.Out...);
X3 = zeros(Float32,R3.Out);
X4 = zeros(Float32,R4.Out);

E0 = zeros(Float32,R1.In...);
E1 = zeros(Float32,R1.Out...);
E2 = zeros(Float32,R2.Out...);
E3 = zeros(Float32,R3.Out);
E4 = zeros(Float32,R4.Out);
O  = zeros(Float32,R4.Out)

# Sorteio:
aux=rand(size(X,1));
tudo=sortperm(aux);
pLimiar = round(Int32,size(X,1)*0.7);
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
		Jteste[ciclo] += sum(E4.^2)/length(E4);
		if (argmax(O)==argmax(X4))
			Ateste[ciclo] += 1;
		end

	end
	
	for n in pTreino
	
		redeIda(n);

		# Medidas de desempenho:
		J[ciclo] += sum(E4.^2)/length(E4);
		if (argmax(O)==argmax(X4))
			A[ciclo] += 1;
		end
		
		redeVolta();
	end


	J[ciclo] = J[ciclo]/length(pTreino);
	A[ciclo] = A[ciclo]/length(pTreino);
	
	Jteste[ciclo] = Jteste[ciclo]/length(pTeste);
	Ateste[ciclo] = Ateste[ciclo]/length(pTeste);
	
	pJ=plot(J);
	pJt=plot!(Jteste);
	
	pA=plot(A);
	pAt=plot!(Ateste);
	
	pz=plot(pJt,pAt, layout=grid(2,1))
	
	display(pz)
	
	println("Ciclo: ", ciclo, " Taxa de acerto (treino): ", 100*round(A[ciclo],digits =4), "%", "\n");
	println("Ciclo: ", ciclo, " Taxa de acerto (teste): ", 100*round(Ateste[ciclo],digits =4), "%", "\n");

end