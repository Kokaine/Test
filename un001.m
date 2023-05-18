clear

load('Example-HalfFilledWaveGuide-mesh1.mat');
% load('Example-HalfFilledWaveGuide-mesh2.mat');

c0 = 299792458;
lam0  =2 ;
k0 = 2*pi/lam0;
a = 0.45789;
epsi_r(1) = 1;
epsi_r(2) = 2.45;
mu_r(1) = 1;
mu_r(2) = 1;
epsi0 = 8.85418781761e-12;
mu0 = 4*pi*1e-7;
% read mesh file 

ipat= mesh.ipat;
xyznode = mesh.xyznode;
% check mesh
Npatch = length(ipat(:,1));
xyznorm = zeros(3,Npatch);
xyzctr = zeros(3,Npatch);
for ii = 1:Npatch 
    p1 = xyznode(:,ipat(ii,1)) + xyznode(:,ipat(ii,2)) + xyznode(:,ipat(ii,3));
    p1 = p1/3;
    e12 = xyznode(:,ipat(ii,2)) - xyznode(:,ipat(ii,1));
    e13 = xyznode(:,ipat(ii,3)) - xyznode(:,ipat(ii,1));
    e12 = [e12; 0];
    e13 = [e13; 0];
    xyzctr(:,ii) = [p1; 0];
    xyznorm(:,ii) = cross(e12,e13)/norm(cross(e12,e13));
end

trimesh(ipat(:,1:3),xyznode(1,:),xyznode(2,:),0*ones(size(xyznode(2,:))),'edgecolor','k');
axis equal 
hold on 
quiver3(xyzctr(1,:),xyzctr(2,:),xyzctr(3,:),xyznorm(1,:),xyznorm(2,:),xyznorm(3,:),0.4);


% assign material propeties  
Npatch = length(ipat(:,1));
Nnode = length(xyznode(1,:));
ipat = [ipat,zeros(Npatch,1)];
for ii = 1:Npatch
    p1 = xyznode(:,ipat(ii,1)) + xyznode(:,ipat(ii,2)) + xyznode(:,ipat(ii,3));
    p1 = p1/3;
    if(p1(1) < a)
        ipat(ii,4) = 2; % 材料类型1 
    else
        ipat(ii,4) = 1; % 材料类型2 
    end
end

indx = find(ipat(:,4) ==2);

trimesh(ipat(indx,1:3),xyznode(1,:),xyznode(2,:),0*ones(size(xyznode(2,:))),'facecolor','r');


close all

% 统计有多少个边； 那些边在PEC边界上，那些点在PEC边界上；
edge = zeros(2, 3*Npatch);  % 初始化边数组
n = 0;  % 计数器

% 构造边数组
for i = 1:Npatch
    n = n + 1;
    edge(:, n) = ipat(i, 1:2)';    
    n = n + 1;
    edge(:, n) = ipat(i, [1, 3])';    
    n = n + 1;
    edge(:, n) = ipat(i, 2:3)';
end

% 删除重复边
validEdges = true(1, n);  % 用于标记有效边的逻辑数组

for i = 1:n
    for j = i + 1:n
        if (edge(1,i) == edge(1,j) && edge(2,i) == edge(2,j)) || (edge(1,i) == edge(2,j) && edge(2,i) == edge(1,j))
            validEdges(j) = false;  % 标记重复边为无效
        end
    end
end

edge = edge(:, validEdges);  % 保留有效边
[~, sides] = size(edge);  % 获取边的数量

% 在边界上的点
Onside = 0;
Inside = 0;
total = 0;
Onside_edge = zeros(2, sides);
Inside_edge = zeros(2, sides);
L = 0.91578; % 矩形长度
H = 0.45789; % 矩形高度
for i = 1:sides
    P1 = edge(1, i);
    P2 = edge(2, i);
    if (xyznode(1, P1) == 0 && xyznode(1, P2) == 0) || (xyznode(2, P1) == 0 && xyznode(2, P2) == 0) || (xyznode(2, P1) == H && xyznode(2, P2) == H) || (xyznode(1, P1) == L && xyznode(1, P2) == L)
        total = total + 1;
        Onside = Onside + 1;
        Onside_edge(:, Onside) = [P1; P2];
    else
        total = total + 1;
        Inside = Inside + 1;
        Inside_edge(:, Inside) = [P1; P2];
    end
end
ns_flag = 1
for i =1:Nnode
    if (xyznode(1,i) == 0) || (xyznode(2,i) == 0) || (xyznode(1,i) == 0.91578) || (xyznode(2,i) == 0.45789)
        bian_node(ns_flag) = i
        ns_flag = ns_flag + 1
    end
end
total_edge = [Onside_edge(:, 1:Onside) Inside_edge(:, 1:Inside)];

tri_edge = zeros(3, Npatch);
for i = 1:Npatch
    for j = 1:sides 
        % check if the edge matches the endpoints of the current triangle
        if (ipat(i,1) == total_edge(1,j) && ipat(i,2) == total_edge(2,j)) || (ipat(i,1) == total_edge(2,j) && ipat(i,2) == total_edge(1,j))
            tri_edge(3,i) = j 
        end
        if (ipat(i,1) == total_edge(1,j) && ipat(i,3) == total_edge(2,j)) || (ipat(i,3) == total_edge(1,j) && ipat(i,1) == total_edge(2,j))
            tri_edge(2,i) = j ;
        end
        if (ipat(i,2) == total_edge(1,j) && ipat(i,3) == total_edge(2,j)) || (ipat(i,2) == total_edge(2,j) && ipat(i,3) == total_edge(1,j))
            tri_edge(1,i) = j ;
        end
    end
end
close all



% 填充FEM 矩阵A，B 
Att=zeros(sides,sides);
Btz=zeros(sides,Nnode);
Bzt=zeros(Nnode,sides);
Btt=zeros(sides,sides);
Bzz=zeros(Nnode,Nnode);

for n=1:1:Npatch
    % basic number
    % point coordinates
    x(1,1:3)=xyznode(1,ipat(n,1:3));
    y(1,1:3)=xyznode(2,ipat(n,1:3));
    % triangle area
    s=abs(x(1)*y(2)+x(2)*y(3)+x(3)*y(1)-x(1)*y(3)-x(2)*y(1)-x(3)*y(2))/2;
    b(1)=(y(2)-y(3))/2;b(2)=(y(3)-y(1))/2;b(3)=(y(1)-y(2))/2;
    c(1)=(x(3)-x(2))/2;c(2)=(x(1)-x(3))/2;c(3)=(x(2)-x(1))/2;
    % edge length
    l(1)=sqrt((x(3)-x(2))^2+(y(3)-y(2))^2);
    l(2)=sqrt((x(3)-x(1))^2+(y(3)-y(1))^2);
    l(3)=sqrt((x(1)-x(2))^2+(y(1)-y(2))^2);
    % gradient
    gradL(1,1:3)=[b(1)/(s) c(1)/s 0];
    gradL(2,1:3)=[b(2)/(s) c(2)/s 0];
    gradL(3,1:3)=[b(3)/(s) c(3)/s 0];
    % other constants
    E=[2.45,1];
    d=[2,3;3,1;1,2];
    
    
    
    deta = eye(3);
    r = zeros(3, 3, 3);
    g = zeros(3, 3);    
    for i = 1:3
        for j = 1:3
            if i ~= j
                deta(i, j) = 0;
            end
            r(i,j,1:3)=cross(gradL(i,1:3),gradL(j,1:3));
            g(i,j)=sum(gradL(i,1:3).*gradL(j,1:3));
        end
    end

    for i=1:3
        for j=1:3
            Q(i,j)=(1+deta(i,j))*s/12;
            P(i,j)=g(i,j)*s;
            T(i,j)=4*s*l(i)*l(j)*sum(r(d(i,1),d(i,2),1:3).*r(d(j,1),d(j,2),1:3));
            R(i,j)=((1+deta(d(i,1),d(j,1)))*g(d(i,1),d(j,1)) - (1+deta(d(i,1),d(j,2)))*g(d(i,2),d(j,1)) - (1+deta(d(i,2),d(j,1)))*g(d(i,1),d(j,2)) + (1+deta(d(i,2),d(j,2)))*g(d(i,2),d(j,2))) *l(i)*l(j)*s/12;
            U(i,j)=l(i)*s*(g(d(i,2),j)-g(d(i,1),j))/3;
        end
    end

    Ttest = ones(3, 3);
    Utest = ones(3, 3);

    for i = 1:3
        itest1 = sign(ipat(n, d(i, 2)) - ipat(n, d(i, 1)));
        for j = 1:3
            itest2 = sign(ipat(n, d(j, 2)) - ipat(n, d(j, 1)));
            Ttest(i, j) = itest1 * itest2;
        end
        Utest(i, 1:3) = itest1;
    end

    U = U .* Utest;
    T = T .* Ttest;
    R = Ttest .* R;

    Aett = T - (pi^2) .* E(1, ipat(n, 4)) * R;
    Betz = U;
    Bezt = Betz';
    Bezz = P - (pi^2) .* E(1, ipat(n, 4)) * Q;
    Bett = R;

    % Att Btt Btz Bzz Bzt need to be assembled based on PEC-filtered results
    for i = 1:3
        for j = 1:3
            Att(tri_edge(i, n), tri_edge(j, n)) = Att(tri_edge(i, n), tri_edge(j, n)) + Aett(i, j);
            Btt(tri_edge(i, n), tri_edge(j, n)) = Btt(tri_edge(i, n), tri_edge(j, n)) + Bett(i, j);
            Btz(tri_edge(i, n), ipat(n, j)) = Btz(tri_edge(i, n), ipat(n, j)) + Betz(i, j);
            Bzz(ipat(n, i), ipat(n, j)) = Bzz(ipat(n, i), ipat(n, j)) + Bezz(i, j);
            Bzt(ipat(n, i), tri_edge(j, n)) = Bzt(ipat(n, i), tri_edge(j, n)) + Bezt(i, j);
        end
    end

end

% 求解广义特征方程
Edge_num = 1:sides
Node_num = 1:Nnode
Edge_num(1:Onside) = []
Node_num(bian_node) = []
Att = Att(Edge_num,Edge_num);
Btt = Btt(Edge_num,Edge_num);
Bzz = Bzz(Node_num,Node_num);
Btz = Btz(Edge_num,Node_num);
Bzt = Bzt(Node_num,Edge_num);
Bzzni=Bzz^(-1);
Bepie=Btz*Bzzni*Bzt-Btt;


% SOLVE!!!!!
[V,D] = eig(Att,Bepie);
K = sort(sqrt(diag(D)));
K(1:3)
% Kc_p = sort(sqrt(eig(Att/Bepie)));

return 

return 
