include("juJuBra.jl") #Biblioteca JuTorch.jl
using DelimitedFiles
using Plots
using JLD2
using Statistics

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

m = mean(X, dims=1) # média de cada coluna, resultado 1×n
s = std(X, dims=1) # desvio padrão de cada coluna, resultado 1×n

X .= (X .- ones(size(X,1), 1) * m) # subtrai a média de cada coluna
X .= X ./ (ones(size(X,1), 1) * s .+ 1e-8)  # divide pelo desvio padrão com eps para evitar divisão por zero

# Gradiente estocástico simples com Adam
alpha = 0.001;


############################################
# Criando a topologia da Rede Neural
############################################

# Primeira Camada (convolutiva) Inicio do Codificador
m1,n1,k1,c1=3,3,1,40; # dimensões dos filtros:
R1 = redeConvolutiva(1/sqrt(m1*n1*k1)*(randn(Float32,m1,n1,k1,c1)), #W
						zeros(1,1,1,c1), #B
						x -> tanh(x), #f(x)
						f -> 1-f^2, #f'(x)
						1, #Stride
						[16,16] #Dimensões da entrada
						);

# Segunda camada (max pooling):
R2 = redeMaxPooling(R1.Out, #dimensões da entrada
                    2, #Stride
                    [2,2], #Dimensões do Pooling
                    );

# Terceira camada (Convolucional):
m3,n3,k3,c3=2,2,40,40; # dimensões dos filtros:
R3 = redeConvolutiva(1/sqrt(m3*n3*k3)*(randn(Float32,m3,n3,k3,c3)), #W
						zeros(1,1,1,c3), #B
						x -> tanh(x), #f(x)
						f -> 1-f^2, #f'(x)
						1, #Stride
						R2.Out #Dimensões da entrada
						);

# Quarta camada (max pooling):
R4 = redeMaxPooling(R3.Out, #dimensões da entrada
                    2, #Stride
                    [2,2], #Dimensões do Pooling
                    );

# Quinta camada (ConvolutivaT):
R5 = redeConvolutivaT(1/sqrt(3*3*40)*randn(Float32,3,3,40,40),
				x -> tanh(x), #f(x)
				f -> 1-f^2, #f'(x)
                2, #Stride
                R4.Out #Dimensões da entrada
                );

# Sexta camada (ConvolutivaT):
R6 = redeConvolutivaT(1/sqrt(3*3*40)*randn(Float32,3,3,40,40),
				x -> tanh(x), #f(x)
				f -> 1-f^2, #f'(x)
                2, #Stride
                R5.Out #Dimensões da entrada
                );

# Sétima camada (Convolucional):
m3,n3,k3,c3=1,1,40,1; # dimensões dos filtros:
R7 = redeConvolutiva(1/sqrt(m3*n3*k3)*(randn(Float32,m3,n3,k3,c3)), #W
						zeros(1,1,1,c3), #B
						x -> tanh(x), #f(x)
						f -> 1-f^2, #f'(x)
						1, #Stride
						R6.Out #Dimensões da entrada
						);


# Ida
function redeIda(n)
    X0 .= Array(reshape(X[n,:],R1.In...)) #X0: entrada da rede');
	idaConvolutiva(R1,X0,X1)
    idaMaxPooling(R2,X1,X2)
	idaConvolutiva(R3,X2,X3);
    idaMaxPooling(R4,X3,X4);
	idaConvolutivaT(R5,X4,X5);
	idaConvolutivaT(R6,X5,X6);
	idaConvolutiva(R7,X6,X7);

    # Cálculo do erro:

    E7 .= (X7 .- X0[1:end-1,1:end-1]);
end

# Volta
function redeVolta()
	voltaConv(R7,X6,X7,E6,E7);
	voltaConvolutivaT(R6,X5,X6,E5,E6);
	voltaConvolutivaT(R5,X4,X5,E4,E5);
    voltaMaxPooling(R4,X3,X4,E3,E4);
    voltaConv(R3,X2,X3,E2,E3);
	voltaMaxPooling(R2,X1,X2,E1,E2)
	voltaConv(R1,X0,X1,E0,E1)
end

#Inicialização da variáveis de treinamento:
X0 = zeros(Float32,R1.In...);
X1 = zeros(Float32,R1.Out...);
X2 = zeros(Float32,R2.Out...);
X3 = zeros(Float32,R3.Out...);
X4 = zeros(Float32,R4.Out...);
X5 = zeros(Float32,R5.Out...);
X6 = zeros(Float32,R6.Out...);
X7 = zeros(Float32,R7.Out...);

E0 = zeros(Float32,R1.In...);
E1 = zeros(Float32,R1.Out...);
E2 = zeros(Float32,R2.Out...);
E3 = zeros(Float32,R3.Out...);
E4 = zeros(Float32,R4.Out...);
E5 = zeros(Float32,R5.Out...);
E6 = zeros(Float32,R6.Out...);
E7 = zeros(Float32,R7.Out...);
O  = zeros(Float32,R7.Out...)

# Sorteio:
aux=rand(size(X,1));
#tudo=sortperm(aux);
tudo = 1:size(X,1);
pTreino = tudo;


############################################
# Medidas de desempenho
############################################
Nciclos = 100;
J=zeros(Float32,Nciclos,1);

for ciclo in 1:Nciclos
	
	for n in pTreino
	
		redeIda(n);

		# Medidas de desempenho:
		J[ciclo] += sum(E7.^2)/length(E7);
		
		redeVolta();
	end


	J[ciclo] = J[ciclo]/length(pTreino);
	
	
	pJ=plot(J, label = "EQM train");
	
	display(pJ)

end