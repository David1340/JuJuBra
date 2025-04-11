include("juJuBra.jl") #Biblioteca JuTorch.jl
using DelimitedFiles
using Plots

# ############################################
# # Leitura de dados
# ############################################

Y=readdlm("datasets/numeros.txt");
X=Y[:,1:256];
L=Y[:,257];

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
m1,n1,k1,c1=3,3,1,10; # dimensões dos filtros:
RC1 = redeConvolutiva(1/sqrt(m1*n1*k1)*(randn(m1,n1,k1,c1)), #W
						zeros(1,1,1,c1), #B
						x -> max(x,0), #f(x) = Relu(x)
						f -> Float64(f > 0), #f'(x)
						1, #Stride
						[16,16] #Dimensões da entrada
						);
dc1 = prod(RC1.Out);
# Segunda camada (densa):
dc2 = 50; # Número de neurônios na primeira camada densa

RD2 = redeDensa(1/sqrt(dc1)*randn(dc2,dc1), # W
                zeros(dc2,1), # B
                x -> 1/(1 + exp(-x)), #f(x)
                f -> f * (1-f)); #df(x)

# Terceira camada (densa):
dc3 = Nclasses;
RD3 = redeDensa(1/sqrt(dc2)*randn(dc3,dc2), # W
                zeros(dc3,1), # B
                x -> 1/(1 + exp(-x)), #f(x)
                f -> f * (1-f)); #df(x)

# Ida
function redeIda(n)
    X0 .= Array(reshape(X[n,:],RC1.In...)) #X0: entrada da rede');

	# Camada densa 1:
	idaConvolutiva(RC1,X0,X1)

    # Camada densa 2:
	idaDensa(RD2,X1,X2);

	# Camada densa 23:
	idaDensa(RD3,X2,X3);

    # Cálculo do erro:
    O .= 0;
    O[convert(Int32,L[n])]=1;

    E3 .= (X3 .- O);
end

# Volta
function redeVolta()

    # Retropropagação pela segunda camada densa:
    voltaDensa(RD3,X2,X3,E2,E3);

    # Retropropagação pela primeira camada densa:
    voltaDensa(RD2,X1,X2,E1,E2);

	voltaConv(RC1,X0,X1,E0,E1)
end

#Inicialização da variáveis de treinamento:
X0 = zeros(Float64,RC1.In...);
X1 = zeros(Float64,RC1.Out...);
X2 = zeros(Float64,RD2.Out);
X3 = zeros(Float64,RD3.Out);

E0 = zeros(Float64,RC1.In...);
E1 = zeros(Float64,RC1.Out...);
E2 = zeros(Float64,RD2.Out);
E3 = zeros(Float64,RD3.Out);
O  = zeros(Float64,RD3.Out)

# Sorteio:
aux=rand(size(X,1));
tudo=sortperm(aux);
pLimiar = round(Int32,size(X,1)*0.7);
pTreino = tudo[1:pLimiar];
pTeste = tudo[pLimiar+1:end];

############################################
# Medidas de desempenho
############################################
Nciclos = 100;
J=zeros(Float64,Nciclos,1);
A = zeros(Float64,Nciclos,1);
Jteste=zeros(Float64,Nciclos,1);
Ateste = zeros(Float64,Nciclos,1)

for ciclo in 1:Nciclos
	for n in pTeste

		redeIda(n);

		# Medidas de desempenho:
		Jteste[ciclo] += sum(E3.^2)/length(E3);
		if (argmax(O)==argmax(X3))
			Ateste[ciclo] += 1;
		end

	end
	
	for n in pTreino
	
		redeIda(n);

		# Medidas de desempenho:
		J[ciclo] += sum(E3.^2)/length(E3);
		if (argmax(O)==argmax(X3))
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
	
	println("Ciclo: ", ciclo, " Taxa de acerto (treino): ", 100*round(A[ciclo],digits =2), "%", "\n");
	println("Ciclo: ", ciclo, " Taxa de acerto (teste): ", 100*round(Ateste[ciclo],digits =2), "%", "\n");

end