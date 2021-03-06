classdef PendulumPlant < SecondOrderSystem
% Defines the dynamics for the Pendulum.
  
  properties
    m = 1;   % kg
    l = .5;  % m
    b = 0.1; % kg m^2 /s
    lc = .5; % m
    I = .25; %m*l^2; % kg*m^2
    g = 9.81; % m/s^2
    
    xG;
    uG;
  end
  
  methods
    function obj = PendulumPlant(b)
      % Construct a new PendulumPlant
      obj = obj@SecondOrderSystem(1,1,true);

      if nargin>0 && ~isempty(b) % accept damping as optional input
        obj.b = b;
      end
      
      obj = setInputFrame(obj,PendulumInput);
      torque_limit = 3;
      obj = setInputLimits(obj,-torque_limit,torque_limit);
      
      obj = setStateFrame(obj,PendulumState);
      obj = setOutputFrame(obj,PendulumState);
      
      obj.xG = Point(getStateFrame(obj),[pi;0]);
      obj.uG = Point(getInputFrame(obj),0);
    end
    
    function qdd = sodynamics(obj,t,q,qd,u)
      % Implement the second-order dynamics
      qdd = (u - obj.m*obj.g*obj.lc*sin(q) - obj.b*qd)/obj.I;
    end
    
    function [f,df,d2f,d3f]=dynamics(obj,t,x,u)
      f=dynamics@SecondOrderSystem(obj,t,x,u);
      if (nargout>1)
        [df,d2f,d3f]= dynamicsGradients(obj,t,x,u,nargout-1);
      end
    end
    
    function [T,U] = energy(obj,x)
      theta = x(1);
      thetadot = x(2);
      T = .5*obj.m*obj.l^2*thetadot^2;
      U = -obj.m*obj.g*obj.l*cos(theta);
    end
    
    function x = getInitialState(obj)
      % Start me anywhere!
      x = randn(2,1);
    end
  end  
  
  methods 
    function [c,V]=balanceLQR(obj)
      Q = diag([10 1]); R = 1;
      if (nargout<2)
        c = tilqr(obj,obj.xG,obj.uG,Q,R);
      else
        if any(~isinf([obj.umin;obj.umax]))
          error('currently, you must disable input limits to estimate the ROA');
        end
        [c,V] = tilqr(obj,obj.xG,obj.uG,Q,R);
        pp = feedback(obj.taylorApprox(0,obj.xG,obj.uG,3),c);
        options.method='levelSet';
        V=regionOfAttraction(pp,V,options);
      end
    end
    
    function [utraj,xtraj]=swingUpTrajectory(obj,options)
      x0 = [0;0]; 
      xf = double(obj.xG);
      tf0 = 4;

      N = 21;
      traj_opt = DircolTrajectoryOptimization(obj,N,[2 6]);
      traj_opt = traj_opt.addStateConstraint(ConstantConstraint(x0),1);
      traj_opt = traj_opt.addStateConstraint(ConstantConstraint(xf),N);
      traj_opt = traj_opt.addRunningCost(@cost);
      traj_opt = traj_opt.addFinalCost(@finalCost);
      traj_init.x = PPTrajectory(foh([0,tf0],[double(x0),double(xf)]));
      
      
      function [g,dg] = cost(dt,x,u);
        R = 10;
        g = (R*u).*u;
        
        if (nargout>1)
          dg = [zeros(1,3),2*u'*R];
        end
      end
      
      function [h,dh] = finalCost(tf,x)
        h = tf;
        if (nargout>1)
          dh = [1, zeros(1,2)];
        end
      end
      
      info=0;
      while (info~=1)
        tic
        [xtraj,utraj,z,F,info] = traj_opt.solveTraj(tf0,traj_init);
        toc
      end
    end
    
    function c=trajectorySwingUpAndBalance(obj)
      [ti,Vf] = balanceLQR(obj);
      Vf = 5*Vf;  % artificially prune, since ROA is solved without input limits

      c = LQRTree(obj.xG,obj.uG,ti,Vf);
      [utraj,xtraj]=swingUpTrajectory(obj);  
      
      Q = diag([10 1]);  R=1;
      [tv,Vswingup] = tvlqr(obj,xtraj,utraj,Q,R,Vf);
      psys = taylorApprox(feedback(obj,tv),xtraj,[],3);
      options.degL1=2;
      Vswingup=sampledFiniteTimeVerification(psys,xtraj.getBreaks(),Vf,Vswingup,options);

      c = c.addTrajectory(xtraj,utraj,tv,Vswingup);
      
      c = setInputFrame(c,c.getInputFrame.constructFrameWithAnglesWrapped([1;0]));
    end
    
    function c=balanceLQRTree(obj)
      Q = diag([10 1]); R = 1;

      options.num_branches=5;
%      options.stabilize=true;
%      options.verify=false;
      options.xs = [0;0];
      options.Tslb = 2;
      options.Tsub = 6;
      options.degL1=4;
      c = LQRTree.buildLQRTree(obj,obj.xG,obj.uG,@()rand(2,1).*[2*pi;10]-[pi;5],Q,R,options);
    end
    
    function animate(obj, traj)
        nPoints = 501;
        t = linspace(traj.tspan(1), traj.tspan(2), nPoints);
        x = eval(traj,t);
        
        theta = pi-x(1,:);
        l1 = 1;
        beta1 = 0.05;
        epsilon = 0.05;
        
        xmin = -l1-epsilon;
        ymin = -l1-epsilon;
        xmax = l1+epsilon;
        ymax = l1+epsilon;
        
        lent = length(t);
        stepSize = 1;
        
        scrsz = get(groot, 'ScreenSize');
        fig = figure(100);
        clf(fig);
        set(fig, 'OuterPosition', [1 scrsz(4)/6 5*scrsz(3)/6 5*scrsz(4)/6]);
        
        R1 = eul2rotm([theta(1), 0, 0]); i1 = R1(1:2,1); j1 = R1(1:2,2);
        p11 = -beta1*i1; p12 = beta1*i1; p13 = p12 + l1*j1; p14 = p11 + l1*j1;
        
        h1 = line([p11(1); p12(1); p13(1); p14(1); p11(1)], [p11(2); p12(2); p13(2); p14(2); p11(2)]);
        h3 = line([0;0],[-2*(l1); 2*(l1)]);
        
        h1.LineWidth = 2;
        h1.Color = 'blue';
        h3.LineStyle = '--'; h3.Color = 'k';
        
        axis([xmin, xmax, ymin, ymax])
        
        line( [xmin; xmax], [ymin; ymin], 'LineWidth', 2, 'Color', [.1 .1 .1], 'LineStyle', '--')
        line( [xmin; xmax], [ymax; ymax], 'LineWidth', 2, 'Color', [.1 .1 .1], 'LineStyle', '--')
        line( [xmin; xmin], [ymin; ymax], 'LineWidth', 2, 'Color', [.1 .1 .1], 'LineStyle', '--')
        line( [xmax; xmax], [ymin; ymax], 'LineWidth', 2, 'Color', [.1 .1 .1], 'LineStyle', '--')
        
        xlabel('x')
        ylabel('y')
        
        
        writerObj = VideoWriter('Pendulum.avi','Motion JPEG AVI');
            writerObj.FrameRate = 60;
            open(writerObj);
            
        for i = 1:stepSize:lent
            
            R1 = eul2rotm([theta(i), 0, 0]); i1 = R1(1:2,1); j1 = R1(1:2,2);
            
            p11 = -beta1*i1; p12 = beta1*i1; p13 = p12 + l1*j1; p14 = p11 + l1*j1;
            
            h1.XData = [p11(1); p12(1); p13(1); p14(1); p11(1)];
            h1.YData = [p11(2); p12(2); p13(2); p14(2); p11(2)];
            
            drawnow
            
            %         if i == 1
            %             keyboard
            %         end
            
            frame = getframe(gca);
                writeVideo(writerObj,frame);
        end
        close(writerObj);
    end

  end

end
