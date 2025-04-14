struct redeDensa
    W::Array{Float32} # permite qualquer número de dimensões
    B::Array{Float32} # permite qualquer número de dimensões
    f::Function # função de ativação
    df::Function # derivada da função de ativação
	In::Int64 #Dimensão da entrada
	Out::Int64 #Dimensão da saída
    m1W::Array{Float32}
    m1B::Array{Float32}    
	m2W::Array{Float32} 
    m2B::Array{Float32}
end

# Construtor auxiliar para criar os parâmetros In e Out de forma automática
function redeDensa(W,B,f,df)
	Out,In = size(W)
	m1W = zeros(Float32,size(W))
	m1B = zeros(Float32,size(B))
	m2W = zeros(Float32,size(W))
	m2B = zeros(Float32,size(B))
	return redeDensa(W,B,f,df,In,Out,m1W,m1B,m2W,m2B)
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

function voltaDensa2(R::redeDensa,X,Y,EX,EY)
    global alpha,beta1,beta2,AdamEpsilon,t
    EY .= EY.*R.df.(Y); # EY = EY .* df(Y) "Pedágio"
	#Cálculo do primeiro e segundo momento para B
	gB = EY;
	R.m1B .= beta1*R.m1B + (1-beta1)*gB;
	R.m2B .= beta2*R.m2B + (1-beta2)*gB.^2;
	m1B = R.m1B/(1-beta1^t); # Corrige o viés
	m2B = R.m2B/(1-beta2^t); # Corrige o viés
	#Atualização de B
    R.B .-= alpha*m1B./(sqrt.(m2B) .+ AdamEpsilon); # B = B - alpha*EY
	#Cálculo do primeiro e segundo momento para B
	gW = EY*X[:]';
	R.m1W .= beta1*R.m1W + (1-beta1)*gW;
	R.m2W .= beta2*R.m2W + (1-beta2)*gW.^2;
	m1W = R.m1W/(1-beta1^t); # Corrige o viés
	m2W = R.m2W/(1-beta2^t); # Corrige o viés
	#Atualização de W
	R.W .-= alpha*m1W./(sqrt.(m2W) .+ AdamEpsilon); # W = W - alpha*EY*X'

    retro =  R.W'*EY; # EX = W'*EY
    EX .= Array(reshape(retro,size(EX)));
end

struct redeConvolutiva 
    W::Array{Float32} # Pesos
    B::Array{Float32} # Bias
    f::Function # função de ativação
    df::Function # derivada da função de ativação
    S::Int64 # stride
	In::Array{Int64} #Dimensões da entrada
	Out::Array{Int64} #Dimensões da saída
	pX::Array{Int64} # apontador
	m1W::Array{Float32}
    m1B::Array{Float32}    
	m2W::Array{Float32} 
    m2B::Array{Float32}
end

# Construtor auxiliar para criar os parâmetros pX e Out de forma automática
function redeConvolutiva(W,B,f,df,S,In)
	pX = apontaConv(In,W,S);
	Out = dimSaidaConv(In,W,S);
	m1W = zeros(Float32,size(W));
	m1B = zeros(Float32,size(B));
	m2W = zeros(Float32,size(W));
	m2B = zeros(Float32,size(B));
	return redeConvolutiva(W,B,f,df,S,In,Out,pX,m1W,m1B,m2W,m2B)
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
    elseif(ndims(R.W) == 4)
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
	if(ndims(X) == 2)
		Mx,Nx = size(X)
		Kx = 1;
	elseif(ndims(X) == 3)
		Mx,Nx,Kx = size(X);
	else
		@warn("Atenção: incosistência de dimensões de X na função VvltaConv!")
	end
	
	if(ndims(R.W) == 3)
		Mw,Nw,C = size(R.W);
		Kw = 1;
	elseif(ndims(R.W) == 4)
		Mw,Nw,Kw,Cw = size(R.W);
	else
		@warn("Atenção: incosistência de dimensões de W na função voltaConv!")
	end
	EY .= EY.*R.df.(Y);
	if(ndims(Y) == 2)
		My,Ny = size(Y);
		Ky = 1;
	elseif(ndims(Y) == 3)
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
	for (index, value) in enumerate(R.pX)
		EX[value] += auxEa[index]
	end

end

function voltaConv2(R::redeConvolutiva,X,Y,EX,EY) #X: entrada, EX: na camada atual, pX: apontador de X, Y: saída, EY: erro da saída
	global alpha,beta1,beta2,AdamEpsilon,t
	if(ndims(X) == 2)
		Mx,Nx = size(X)
		Kx = 1;
	elseif(ndims(X) == 3)
		Mx,Nx,Kx = size(X);
	else
		@warn("Atenção: incosistência de dimensões de X na função VvltaConv!")
	end
	
	if(ndims(R.W) == 3)
		Mw,Nw,C = size(R.W);
		Kw = 1;
	elseif(ndims(R.W) == 4)
		Mw,Nw,Kw,Cw = size(R.W);
	else
		@warn("Atenção: incosistência de dimensões de W na função voltaConv!")
	end
	EY .= EY.*R.df.(Y);
	if(ndims(Y) == 2)
		My,Ny = size(Y);
		Ky = 1;
	elseif(ndims(Y) == 3)
		My,Ny,Ky = size(Y);
	else
		@warn("Atenção: incosistência de dimensões de Y na função voltaConv!")
	end

	auxY = My*Ny; #length de cada canal de Y
	auxW = Mw*Nw*Kw; #length de cada Kernel
	auxEa = zeros(size(R.pX))
	for canal in 1:Cw
		#Cálculo do primeiro e segundo momento para B
		gB = sum(EY[auxY*(canal -1)+1:auxY*canal]);
		R.m1B[canal] = beta1*R.m1B[canal] + (1-beta1)*gB;
		R.m2B[canal] = beta2*R.m2B[canal] + (1-beta2)*gB.^2;
		m1B = R.m1B[canal]/(1-beta1^t); # Corrige o viés
		m2B = R.m2B[canal]/(1-beta2^t); # Corrige o viés
		R.B[canal] += - alpha*m1B./(sqrt.(m2B) .+ AdamEpsilon); #Atualiza o bias da camada atual

		#Cálculo do primeiro e segundo momento para W
		gW = X[R.pX]*EY[auxY*(canal -1)+1:auxY*canal];
		R.m1W[auxW*(canal -1)+1:auxW*canal] = beta1*R.m1W[auxW*(canal -1)+1:auxW*canal] + (1-beta1)*gW;
		R.m2W[auxW*(canal -1)+1:auxW*canal] = beta2*R.m2W[auxW*(canal -1)+1:auxW*canal] + (1-beta2)*gW.^2;
		m1W = R.m1W[auxW*(canal -1)+1:auxW*canal]/(1-beta1^t); # Corrige o viés
		m2W = R.m2W[auxW*(canal -1)+1:auxW*canal]/(1-beta2^t); # Corrige o viés
		R.W[auxW*(canal -1)+1:auxW*canal] .+= -alpha*m1W./(sqrt.(m2W) .+ AdamEpsilon); #Atualiza os pesos da camada atual

		auxEa += R.W[auxW*(canal -1)+1:auxW*canal]*EY[auxY*(canal -1)+1:auxY*canal]'; #Atualiza o erro da camada anterior
	end
	EX .= zeros(size(X)); #Zera o erro da camada atual
	for (index, value) in enumerate(R.pX)
		EX[value] += auxEa[index]
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
	elseif(length(In) == 3)
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
		@warn("Atenção: número de canais da camada convolutiva diferente do número de canais do kernel  (apontaConv)!")
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

struct redeMaxPooling
	In::Array{Int64} #Dimensões da entrada
	S::Int64 # stride
	W::Array{Int64} # dimensões do Pooling
	Out::Array{Int64} #Dimensões da saída
	pX::Array{Int64} # apontador
	pX2::Array{Int64} # apontador
end

# Construtor auxiliar para criar os parâmetros pX, pX2 e Out de forma automática

function redeMaxPooling(In,S,W)
	pX = apontaPooling(In,W,S);
	Out = dimSaidaPooling(In,W,S);
	pX2 = zeros(prod(Out),1)
	return redeMaxPooling(In,S,W,Out,pX,pX2)
end

function idaMaxPooling(R::redeMaxPooling,X,Y)
    if(ndims(X) == 2)
        Mx,Nx = size(X)
        Kx = 1;
    elseif(ndims(X) == 3)
        Mx,Nx,Kx = size(X);
    else
        @warn("Atenção: incosistência de dimensões de X na função idaMaxPooling")
    end

    auxY = prod(size(Y)); #length de cada canal de Y
    R.pX2 .= zeros(size(R.pX2)); #zera o apontador do maxPooling
	for i in 1:auxY	 
		indices = R.pX[:,i];
		valoresX = X[indices];
		max_idx = findmax(valoresX); # retorna (valor, índice)
        Y[i] = max_idx[1];
		R.pX2[i] = indices[max_idx[2]]; #pX2: apontador do maxPooling	
    end  

end

function voltaMaxPooling(R::redeMaxPooling,X,Y,EX,EY)
	if(ndims(X) == 2)
		Mx,Nx = size(X)
		Kx = 1;
	elseif(ndims(X) == 3)
		Mx,Nx,Kx = size(X);
	else
		@warn("Atenção: incosistência de dimensões de X na função voltaMaxPooling")
	end
	

	if(ndims(Y) == 2)
		My,Ny = size(Y);
		Ky = 1;
	elseif(ndims(Y) == 3)
		My,Ny,Ky = size(Y);
	else
		@warn("Atenção: incosistência de dimensões de Y na função voltaMaxPooling")
	end

	auxY = My*Ny*Ky; #length de y
	EX .= zeros(size(EX)) #zera o erro da camada atual
	for i in 1:auxY
		EX[R.pX2[i]] +=  EY[i];
	end

end

function dimSaidaPooling(In,W,S)
	if(length(In) == 2)
		Mx,Nx = In
		Kx = 1;
	elseif(length(In) == 3)
		Mx,Nx,Kx = In;
	else
		@warn("Atenção: incosistência de dimensões de X na função dimSaidaPooling")
	end
	if(length(W) == 2)
		Mw,Nw = W;
	else
		@warn("Atenção: incosistência de dimensões de W na função dimSaidaPooling")
	end

	My=length(1:S:Mx-Mw+1);
	Ny=length(1:S:Nx-Nw+1);
	Ky=Kx;
	return [My,Ny,Ky]
end

function apontaPooling(In,W,S)  
	if(length(In) == 2)
		Mx,Nx = In
		Kx = 1;
	elseif(length(In) == 3)
		Mx,Nx,Kx = In;
	else
		@warn("Atenção: incosistência de dimensões de In função apontaConv!")
	end

	if(length(W) == 2)
		Mw,Nw = W;
	else
		@warn("Atenção: incosistência de dimensões de W na função apontaPooling")
	end
	 
	L = Kx*length(1:S:Mx-Mw+1)*length(1:S:Nx-Nw+1); #número de convoluções
	pConv = zeros(Int64,Mw*Nw,L); 
	aux = reshape(1:Mx*Nx*Kx,Mx,Nx,Kx); #Dada as dimensões de um matriz A cria uma matriz B cujos valores é os índices linerares da matriz A
	cont = 1;
	for k in 1:Kx 
		for n in 1:S:Nx-Nw+1
			for m in 1:S:Mx-Mw+1
				pConv[:,cont] = aux[m:m+Mw-1,n:n+Nw-1,k][:];
				cont += 1;
			end
		end
	end
	return pConv
end

function redeGenericaIda(R::Union{redeDensa, redeConvolutiva,redeMaxPooling},X,Y)
	if isa(R,redeDensa)
		idaDensa(R,X,Y)
	elseif isa(R,redeConvolutiva)
		idaConvolutiva(R,X,Y)
	elseif isa(R,redeMaxPooling)
		idaMaxPooling(R,X,Y)
	else
		@warn("Atenção: tipo de rede não reconhecido na função redeGenericaVolta")
	end
	
end



function redeGenericaVolta(R::Union{redeDensa, redeConvolutiva,redeMaxPooling},X,Y,EX,EY)
	if isa(R,redeDensa)
		voltaDensa(R,X,Y,EX,EY)
	elseif isa(R,redeConvolutiva)
		voltaConv(R,X,Y,EX,EY)
	elseif isa(R,redeMaxPooling)
		voltaMaxPooling(R,X,Y,EX,EY)
	else
		@warn("Atenção: tipo de rede não reconhecido na função redeGenericaVolta")
	end
	
end

function redeGenericaVolta2(R::Union{redeDensa, redeConvolutiva,redeMaxPooling},X,Y,EX,EY)
	if isa(R,redeDensa)
		voltaDensa2(R,X,Y,EX,EY)
	elseif isa(R,redeConvolutiva)
		voltaConv2(R,X,Y,EX,EY)
	elseif isa(R,redeMaxPooling)
		voltaMaxPooling(R,X,Y,EX,EY)
	else
		@warn("Atenção: tipo de rede não reconhecido na função redeGenericaVolta")
	end
	
end