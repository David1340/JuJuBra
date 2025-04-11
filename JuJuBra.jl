mutable struct redeDensa
    W::Array{Float64} # permite qualquer número de dimensões
    B::Array{Float64} # permite qualquer número de dimensões
    f::Function # função de ativação
    df::Function # derivada da função de ativação
	In::Int64 #Dimensão da entrada
	Out::Int64 #Dimensão da saída
end
# Construtor auxiliar para criar os parâmetros In e Out de forma automática
function redeDensa(W,B,f,df)
	Out,In = size(W)
	return redeDensa(W,B,f,df,In,Out)
end

function idaDensa(R::redeDensa,X,Y)
    Y .= R.f.(R.W*X[:]+R.B); # Y = f(W*X + B)
end

function voltaDensa(R::redeDensa,X,Y,EX,EY)
    global alpha
    EY .= EY.*R.df.(Y); # EY = EY .* df(Y) "Pedágio"
    R.B .-= alpha*EY; # B = B - alpha*EY
    retro =  R.W'*EY; # EX = W'*EY
    R.W .-= alpha*EY*X[:]'; # W = W - alpha*EY*X'
    EX .= Array(reshape(retro,size(EX)));
end


mutable struct redeConvolutiva 
    W::Array{Float64} # Pesos
    B::Array{Float64} # Bias
    f::Function # função de ativação
    df::Function # derivada da função de ativação
    S::Int64 # stride
	In::Array{Int64} #Dimensões da entrada
	Out::Array{Int64} #Dimensões da saída
	pX::Array{Int64} # apontador
end

# Construtor auxiliar para criar os parâmetros pX e Out de forma automática
function redeConvolutiva(W,B,f,df,S,In)
	pX = apontaConv(In,W,S);
	Out = dimSaidaConv(In,W,S);
	return redeConvolutiva(W,B,f,df,S,In,Out,pX)
end

function idaConvolutiva(R::redeConvolutiva,X,Y)
    if(length(size(X)) == 2)
        Mx,Nx = size(X)
        Kx = 1;
    elseif(length(size(X)) == 3)
        Mx,Nx,Kx = size(X);
    else
        @warn("Atenção: incosistência de dimensões de X na função idaConvolutiva")
    end

    if(length(size(R.W)) == 3)
        Mw,Nw,C = size(R.W);
        Kw = 1;
    elseif(length(size(R.W)) == 4)
        Mw,Nw,Kw,Cw = size(R.W);
    else
        @warn("Atenção: incosistência de dimensões de W na função idaConvolutiva")
    end

    My,Ny,Ky = size(Y);  #Ky == Cw
    
    auxW = Mw*Nw*Kw; #length de cada Kernel
    auxY = My*Ny; #length de cada canal de Y
    
    for canal in 1:Ky	 
        Y[auxY*(canal-1)+1:auxY*canal] .= R.f.(X[R.pX]'*R.W[auxW*(canal-1)+1:auxW*canal] .+ R.B[canal]); 
    end   
end

function voltaConv(R::redeConvolutiva,X,Y,EX,EY) #X: entrada, EX: na camada atual, pX: apontador de X, Y: saída, EY: erro da saída
	global alpha
	if(length(size(X)) == 2)
		Mx,Nx = size(X)
		Kx = 1;
	elseif(length(size(X)) == 3)
		Mx,Nx,Kx = size(X);
	else
		@warn("Atenção: incosistência de dimensões de X na função VvltaConv!")
	end
	
	if(length(size(R.W)) == 3)
		Mw,Nw,C = size(R.W);
		Kw = 1;
	elseif(length(size(R.W)) == 4)
		Mw,Nw,Kw,Cw = size(R.W);
	else
		@warn("Atenção: incosistência de dimensões de W na função voltaConv!")
	end
	EY .= EY.*R.df.(Y);
	if(length(size(Y)) == 2)
		My,Ny = size(Y);
		Ky = 1;
	elseif(length(size(Y)) == 3)
		My,Ny,Ky = size(Y);
	else
		@warn("Atenção: incosistência de dimensões de Y na função voltaConv!")
	end

	auxY = My*Ny; #length de cada canal de Y
	auxW = Mw*Nw*Kw; #length de cada Kernel
	auxEa = zeros(size(R.pX))
	for canal in 1:Cw

		R.B[canal] += -alpha*sum(EY[auxY*(canal -1)+1:auxY*canal]); #Atualiza o bias da camada atual

		R.W[auxW*(canal -1)+1:auxW*canal] .+= -alpha*X[R.pX]*EY[auxY*(canal -1)+1:auxY*canal]; #Atualiza os pesos da camada atual
		auxEa += R.W[auxW*(canal -1)+1:auxW*canal]*EY[auxY*(canal -1)+1:auxY*canal]'; #Atualiza o erro da camada anterior
	end
	EX .= zeros(size(X)); #Zera o erro da camada atual
	for i in unique(R.pX)
		EX[i] = sum(auxEa[R.pX .== i]); #Atualiza o erro da camada anterior
	end

end

function dimSaidaConv(In,W,S)
	if(length(In) == 2)
		Mx,Nx = In
		Kx = 1;
	elseif(length(In) == 3)
		Mx,Nx,Kx = In;
	else
		@warn("Atenção: incosistência de dimensões de X na função dimSaidaConv!")
	end
	if(length(size(W)) == 3)
		Mw,Nw,C = size(W);
		Kw = 1;
	elseif(length(size(W)) == 4)
		Mw,Nw,Kw,C = size(W);
	else
		@warn("Atenção: incosistência de dimensões de W na função dimSaidaConv!")
	end
	if Kx != Kw
	  @warn("Atenção: número de canais da camada convolutiva diferente do número de canais do kernel  (dimSaidaConv)!")
	end
	My=length(1:S:Mx-Mw+1);
	Ny=length(1:S:Nx-Nw+1);
	Ky=C;
	return [My,Ny,Ky]
end

function apontaConv(In,W,S)  
	if(length(In) == 2)
		Mx,Nx = In
		Kx = 1;
	elseif(size(In) == 3)
		Mx,Nx,Kx = In;
	else
		@warn("Atenção: incosistência de dimensões de X na função apontaConv!")
	end

	if(length(size(W)) == 3)
		Mw,Nw,C = size(W);
		Kw = 1;
	elseif(length(size(W)) == 4)
		Mw,Nw,Kw,C = size(W);
	else
		@warn("Atenção: incosistência de dimensões de W na função apontaConv!")
	end

	if Kx != Kw
		@warn("Atenção: número de canais de canais da convolutiva diferente do número de canais do kernel  (apontaConv)!")
	end 
	 
	L = length(1:S:Mx-Mw+1)*length(1:S:Nx-Nw+1); #número de convoluções
	pConv = zeros(Int64,Mw*Nw*Kw,L); 
	aux = reshape(1:Mx*Nx*Kx,Mx,Nx,Kx); #Dada as dimensões de um matriz A cria uma matriz B cujos valores é os índices linerares da matriz A
	cont = 1;
	for m in 1:S:Mx-Mw+1
	  for n in 1:S:Nx-Nw+1
		pConv[:,cont] = aux[m:m+Mw-1,n:n+Nw-1,:][:];
		cont += 1;
	  end
	end
	return pConv
end

mutable struct redePooling
    pX::Array{Int64} # apontador
    p::Array{Int64} # dimensões do padding
end