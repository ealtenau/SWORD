function [S,D,Wavelength] = SinuosityMinAreaVarMinReach(Latitude,Longitude,Width,lambda,MinReachLen,findlambda,DebugPlot)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    %finds the UTMZone that best fit the Segment
    if length(Latitude)<3
        S=ones(size(Latitude));
        D=zeros(size(Latitude));
        Wavelength=zeros(size(Latitude));
        return
    end
    Zone=utmzone(Latitude,Longitude);
    %find most appropriate geoid for the segment
    %[ellipsoid,names] = utmgeoid(Zone); %only needed if we decide to not
    %project data, but work with great arcs in the function distance
    utmstruct=defaultm('utm'); %configures utm structure
    utmstruct.zone = Zone;
    utmstruct.geoid = wgs84Ellipsoid;
    utmstruct = defaultm(utmstruct);
    %projects the points into the appropriate utm zone
    [X,Y] = mfwdtran(utmstruct,Latitude,Longitude);
    X=smooth(X); %use moving average with span = 5 points to remove the kinks due to pixels
    Y=smooth(Y);
    n=length(X);
    D=nan(size(X));
    D(1)=0;
    Wavelength=nan(size(X));
    for i=2:n,
        D(i)=D(i-1)+ sqrt( (X(i)-X(i-1))^2 + (Y(i)-Y(i-1))^2 );
    end
    if findlambda==1&&length(D)>20
        %instead of assuming a relationship between wavelength and river
        %width, compute sinuosity for each bend.
        Dx=X(2:end)-X(1:end-1);
        Dy=Y(2:end)-Y(1:end-1);
        Product=nan(size(D));
        Product(2:end-1)=Dx(1:end-1).*Dy(2:end)-Dx(2:end).*Dy(1:end-1);
        Product(end)=Product(end-1);
        Product(1)=Product(2);
        Bound=nan(ceil(length(Dx)/2)+1,1);
        Bound(1)=1;
        numbbound=2;
        howmany=0;
        for count=2:length(Product)-1
            Base=sqrt((X(count+1)-X(count-1))^2+(Y(count+1)-Y(count-1))^2);
            height=Product(count)/(2*Base);
            width=4;
            while abs(height)<30&&width<30
                if count-floor(width/2)>0
                    x1=X(count-floor(width/2));
                    y1=Y(count-floor(width/2));
                else
                    x1=X(1);
                    y1=Y(1);
                end
                if count+floor(width/2)<length(X)+1
                    x2=X(count+floor(width/2));
                    y2=Y(count+floor(width/2));
                else
                    x2=X(end);
                    y2=Y(end);
                end
                Dxup=X(count)-x1;
                Dxdo=x2-X(count);
                Dyup=Y(count)-y1;
                Dydo=y2-Y(count);
                prod=Dxup*Dydo-Dxdo*Dyup;
                Base=sqrt((x2-x1)^2+(y2-y1)^2);
                height=prod/(2*Base);
                howmany=howmany+1;
                Product(count)=prod;
                width=width+2;
            end
        end
        for count=2:length(Product)-1
            if Product(count)*Product(count+1)<0
                %compute the height of the paralelogram in Product(count+1)
                %to see if the height is significant
                Bound(numbbound)=count+1;
                numbbound=numbbound+1;    
            end
        end
        %impose minimum arc-length requirement
        Bound(numbbound)=length(X);           
        [Bound] = MergeShortReachesVarMin(Bound(1:numbbound),D,Product,Width);
        numbbound=length(Bound);
        if DebugPlot==1
            figure
            plot(X/1000,Y/1000)
            hold on
            Bound=Bound(1:numbbound);
            plot(X(Bound)/1000,Y(Bound)/1000,'o','MarkerEdgeColor','r',...
                           'MarkerFaceColor','r',...
                           'MarkerSize',6)
            xlabel('Easting (km)')
            ylabel('Northing (km)')
            legend('River centerline','Inflection points')
            axis equal
        end
        if MinReachLen>0
            AveProd=nan(length(Bound)-1,1);
            %recalculate direction of curvature for each "reach"
            for cbound=2:length(Bound)
                AveProd(cbound-1)=mean(Product(Bound(cbound-1):Bound(cbound)-1));
            end
            BoundToRemove=nan(size(Bound));
            numbboundRemove=1;
            for cbound=1:length(AveProd)-1
                if AveProd(cbound)*AveProd(cbound+1)>0
                    %there is no longer a sign reversal, so remove the
                    %boundary
                    BoundToRemove(numbboundRemove)=cbound+1;
                    numbboundRemove=numbboundRemove+1;    
                end              
            end
            numbboundRemove=numbboundRemove-1;
            NewBound=nan(length(Bound)-numbboundRemove,1);
            curbound=1;
            incbound=1;
%             for cboundremove=1:numbbound
            for cboundremove=1:numbboundRemove
                %remove boundaries
                while curbound<BoundToRemove(cboundremove)
                    NewBound(incbound)=Bound(curbound);
                    curbound=curbound+1;
                    incbound=incbound+1;
                end
                curbound=curbound+1;
            end
            curbound=length(Bound);
            incbound=length(NewBound);
            if numbboundRemove>0       
                while numbboundRemove>0&&curbound>BoundToRemove(numbboundRemove)
                    NewBound(incbound)=Bound(curbound);
                    curbound=curbound-1;
                    incbound=incbound-1;
                end
                Bound=NewBound;
                numbbound=length(Bound);
            end
        end
    else
        lambda=min(lambda,2*(D(end)-D(1))+1); %ensures that something is calculated for tiny segments
    end
    
        %% sinuosity calculation
    S=zeros(size(D));
    if findlambda==1&&length(D)>20
        %this method finds changes in centerline direction to identify half
        %wavelenghts used to calculate sinuosity.
        for count=1:numbbound-1
            Ds=D(Bound(count+1))-D(Bound(count)); %Arc length for the bend 
            De=sqrt((X(Bound(count))-X(Bound(count+1)))^2+(Y(Bound(count))-Y(Bound(count+1)))^2);%distance between the bend's endpoints
            S(Bound(count):Bound(count+1))=Ds/De;
            Wavelength(Bound(count):Bound(count+1))=De*2;
        end
        S(end)=S(end-1);
        Wavelength(end)=Wavelength(end-1);
    else      
        for i=1:n,
            jfwd=D>=D(i) & D<D(i)+lambda/2;
            jbak=D<D(i) & D>D(i)-lambda/2;
            j=jfwd | jbak ;

            x=X(j); y=Y(j); d=D(j);

            De=sqrt( (x(1)-x(end))^2 + (y(1)-y(end))^2 );
            Dr=d(end)-d(1);

            S(i)=Dr/De;
            Wavelength(i)=De*2;
        end
        S(end)=S(end-1);
        Wavelength(end)=Wavelength(end-1);
    end
    if MinReachLen>0
        if DebugPlot==1&&exist('Bound','var')
            figure
            plot(X/1000,Y/1000)
            hold on
            title('Filtered inflection points')
            plot(X(Bound)/1000,Y(Bound)/1000,'o','MarkerEdgeColor','r',...
                           'MarkerFaceColor','r',...
                           'MarkerSize',6)
            xlabel('Easting (km)')
            ylabel('Northing (km)')
            legend('River centerline','Inflection points')
            axis equal
        end
    end
end

