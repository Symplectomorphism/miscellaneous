% Written by Aykut
classdef PendulumControl < DrakeSystem
  
  properties
    p  % plant
  end
  
  methods
    function obj = PendulumControl(plant)
      obj = obj@DrakeSystem(0,0,2,1,true,true);
      
      obj.p = plant;
      obj = obj.setInputFrame(plant.getStateFrame);
      %         obj = obj.setInputFrame(plant.getOutputFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
      
    end
    
    function u = output(obj,t,~,x)
        x1 = x(1); x2 = x(2);
        
        cond = 0.461985E1.*x2.^2+0.2166E2.*cos(x1)+(-0.249902E1).*x2.^2.*cos(x1)+(-0.168646E3).*cos(x1).^2+(-0.21245E1).*x2.^2.*cos(x1).^2+(-0.576002E2).*x2.*sin(x1)+0.172026E2.*x2.*cos(x1).*sin(x1)+0.306446E2.*x2.*cos(x1).^2.*sin(x1)+(-0.24891E0).*x2.^2.*sin(x1).^2+(-0.102653E1).*cos(x1).*sin(x1).^2+(-0.53359E0).*x2.^2.*cos(x1).*sin(x1).^2+(-0.796217E2).*cos(x1).^2.*sin(x1).^2+(-0.627774E0).*x2.^2.*cos(x1).^2.*sin(x1).^2;
        
        if cond < -146.692
            u = 0.87682E1.*x2+(-0.57912E1).*x2.*cos(x1)+(-0.4249E1).*x2.*cos(x1).^2+(-0.56625E2).*sin(x1)+0.1762E2.*cos(x1).*sin(x1)+0.30846E2.*cos(x1).^2.*sin(x1)+(-0.97302E0).*x2.*cos(x1).*sin(x1).^2+(-0.13343E1).*x2.*cos(x1).^2.*sin(x1).^2;
%             disp('I am here')
        else
            u = (-0.4715E0).*x2+(-0.79316E0).*x2.*cos(x1)+0.97517E0.*sin(x1)+0.41743E0.*cos(x1).*sin(x1)+0.20138E0.*cos(x1).^2.*sin(x1)+0.49782E0.*x2.*sin(x1).^2+0.9416E-1.*x2.*cos(x1).*sin(x1).^2+(-0.78752E-1).*x2.*cos(x1).^2.*sin(x1).^2;
%             disp('I am there')
        end
        u = -u;
        
%         fprintf(['u = ', num2str(u), '\n']);
        
%         if u >= 0.2
%             u = 0.2;
%         elseif u <= -0.2
%             u = -0.2;Pe
%         end
        
%         fprintf(['time = ', num2str(t), '\n']);
    end
    
  end
  
  methods (Static)
    function run(animFlag)
      p = PendulumPlant();
      c = PendulumControl(p);
%       v = CartPoleVisualizer(cp);
      sys = feedback(p,c);
      
      fig = figure(101);
      phasePortrait(sys, fig);
      
      x0 = [-0.2+0.4*rand; -1+2*rand];
      x0 = [0.05; 0.14];
%       x0 = [0; 60];
      traj = simulate(sys,[0 10], x0);
      t = linspace(0,10,1001);
      x = eval(traj,t);
      figure(1), clf
      plot(t, x);
      if nargin && animFlag
          p.animate(traj)
      end
    end
  end
end
