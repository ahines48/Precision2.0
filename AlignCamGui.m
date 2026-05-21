function [proj_user_points, P_final, ]=AlignCamGui(K, initial_pos, initial_targ, back_image, verts, faces, points, extra_faces, reduce_val, second_image, mriData)
% [created by: Amelia Hines]
% [created on: 06/30/2025]
% [last edited: 07/02/2025]
% K: Camera intrinsic matrix 3x3
% initial_pos: Initial camera position 1x3 
% initial_targ: Initial Camera target 1x3
% back_image: Main background image 
% verts: vertices of a mesh 3xN please!
% faces: corresponding Trianglular faces 
% points: optional argument but if ur using also 3xN
% extra_faces: optional argument, allows user to treat points as vertices
% and import another mesh thats moveable
% reduce_val: Reduces the number of vertices used to generate patches, could speed
% stuff up if rendering is slow specify val from 0 to 1 to be % of vertices kept
% second_image: also optional, same as background image, please dont just put a path
% mriData: Boolean asking if you are putting in an stl generated from our
% MRI data pipeline, if True, an additional rotation and translation are applied to
% the mesh vertices
% MIGHT NEED TO APPLY THE ROTATION AND TRANSLATION FROM MRIDATA THINGY TO
% PROJECTED POINTS! NOT SURE THO, DIDN'T TEST IT. PROLLY NEED TO THO NGL
% (will test soon prolly)


    %% Make Extra Points and Face Reduction Optional
    if nargin < 7 || isempty(points),points = zeros(3,0,'like',K);end
    if nargin < 8 || isempty(extra_faces),extra_faces = [];end
    if nargin < 9 || isempty(reduce_val),reduce_val = 1;end
    if nargin < 10 || isempty(second_image), second_image = [];end
    if nargin < 11 || isempty(mriData), mriData = false;end
    %% Rotate and Move Vertices if mriData True
    if mriData
        R_fix_model = [0 1 0; -1 0 0; 0 0 1];
        move_model = [256 15 60];
        verts = (verts' * R_fix_model)+move_model;
        verts = verts'; % If getting mismatch error comment out this line
    end
    %% Load Stuff and Make Homogenous
    % Convert to homogenous coords
    verts_hom = [verts; ones([1,size(verts,2)])]; 
    points_hom = [points; ones([1,size(points,2)])];

    %% Create Initial Rotation Matrix
    forward_dir_init = (initial_targ-initial_pos)/norm(initial_targ-initial_pos);
    side_dir_world = [1,0,0];
    up_dir_init = cross(forward_dir_init,side_dir_world);
    R_wc_init = [side_dir_world(1), up_dir_init(1), forward_dir_init(1);
                 side_dir_world(2), up_dir_init(2), forward_dir_init(2);
                 side_dir_world(3), up_dir_init(3), forward_dir_init(3)];
    R_cw_init = R_wc_init';
    q_cw_init = quaternion(R_cw_init, 'rotmat', 'frame');
    q_wc_init = quaternion(R_wc_init, 'rotmat', 'frame');
    %% Create Initial Projection
    t_init = -R_cw_init*initial_pos';
    L_init = [R_cw_init, t_init];
    P_init = K*L_init;
    
    proj_verts = P_init*verts_hom;
    proj_points = P_init*points_hom;

    % Divide by Z-axis (distance)
    proj_verts_x = proj_verts(1,:)./proj_verts(3,:);
    proj_verts_y = proj_verts(2,:)./proj_verts(3,:);    

    proj_points_x = proj_points(1,:)./proj_points(3,:);
    proj_points_y = proj_points(2,:)./proj_points(3,:);

    %% Create Initial Figure
    % Set Up Vals for spacing
    gui_width = 300;
    gui_height = 800;
    plot_width = size(back_image, 1);
    plot_height = size(back_image, 2);
    imgSize = size(back_image);
    fig = uifigure('Name', 'Align Cam Gui', 'NumberTitle', 'off', ...
             'Position', [100 100 1500 800],'MenuBar', 'none');
    ax = uiaxes(fig, 'Position',[gui_width,0,900,787]);
    bg = imshow(back_image, 'Parent', ax);
    hold(ax, 'on');
    hPatch = patch(ax,'Faces', faces, 'Vertices', [proj_verts_x', proj_verts_y',zeros(size(proj_verts_x'))], ...
                    'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [.1 .1 .1], 'FaceAlpha', 0.5, 'EdgeAlpha', .1);
    if ~isempty(extra_faces)
        hScatter = patch(ax, 'Faces', extra_faces, 'Vertices', [proj_points_x', proj_points_y', zeros(size(proj_verts_x'))], ...
                            'FaceColor', [0.5 0.5 0.5], 'EdgeColor', [.1 .1 .1], 'FaceAlpha', 0.5, 'EdgeAlpha', .1);
    else
        hScatter = scatter(ax, proj_points_x, proj_points_y, 20, 'r', 'filled');
    end
    hManual = scatter(ax, nan, nan, 30, 'g', 'filled');
    setappdata(fig,'hManual',hManual);
    setappdata(fig,'manual_points', []); 
    axis image;
    xlim(ax, [0,imgSize(1)]);
    ylim(ax, [0, imgSize(2)]);
    hold(ax, 'off');

    % Make sure Fig is big enough
    fig_pos = get(fig, 'Position');
    
    % Add a function call for marking optode locs on the figure
    fig.WindowButtonDownFcn = @(src,event) markPoint(src,event);

    % Store Important stuff in fig app data for easy call
    setappdata(fig,'ax',ax);
    setappdata(fig,'hPatch',hPatch);
    setappdata(fig,'hScatter',hScatter);
    setappdata(fig,'extra_faces',extra_faces);
    setappdata(fig,'bgHandle',bg);          
    setappdata(fig,'isMainShowing',true)
    setappdata(fig,'main_image',back_image);
    setappdata(fig,'alt_image', second_image);
    setappdata(fig,'K', K);
    setappdata(fig,'PointsHom', points_hom);
    setappdata(fig,'VertsHom', verts_hom);
    setappdata(fig,'faces',faces);
    setappdata(fig,'image', image);
    setappdata(fig,'initial_pos', initial_pos);
    setappdata(fig,'camPos_current', initial_pos);
    setappdata(fig,'initial_targ', initial_targ);
    setappdata(fig,'R_cw_init', R_cw_init);
    setappdata(fig,'R_cw_current', R_cw_init);
    setappdata(fig,'q_cw_init', q_cw_init);
    setappdata(fig,'q_wc_init', q_wc_init);
    setappdata(fig, 'mriData', mriData);

    % Create a Control Panel 
    control_panel = uipanel(fig, 'Title', 'Camera Pose Controls', ...
                        'Units', 'pixels', 'Position', [fig_pos(3)-gui_width fig_pos(4)-gui_height gui_width gui_height], ...
                        'FontSize', 12);
    y_offset = control_panel.Position(4) - 60; % How far Down from top
    slider_height = 25;
    label_width = 50;
    value_width = 60;
    gap_height = 90;

    slider_width = control_panel.Position(3) - label_width - value_width-60;
    
    %% Set Up Sliders for Rotations
    R_euler = rotm2eul(R_cw_init, 'YXZ');
    yaw_deg_init = rad2deg(R_euler(1));
    pitch_deg_init = rad2deg(R_euler(2));
    roll_deg_init = rad2deg(R_euler(3));

    % Yaw Slider and Text
    uicontrol(control_panel, 'Style', 'text', 'String', 'Yaw:', ...
              'Position', [10 y_offset label_width slider_height], 'HorizontalAlignment', 'left');

    slider_yaw = uislider(control_panel, 'Value', yaw_deg_init, ...
                         'Position', [10 + label_width y_offset slider_width slider_height], ...
                         'Limits', [yaw_deg_init-22.5 yaw_deg_init+22.5]);
    slider_yaw.ValueChangedFcn = @(s,e) updateFigure(fig);

    setappdata(fig,'slider_yaw',slider_yaw);

    text_yaw = uitextarea(control_panel,'Value', num2str(yaw_deg_init, '%.2f'), ...
                       'Position', [30 + label_width + slider_width y_offset value_width slider_height]);

    setappdata(fig, 'text_yaw', text_yaw);

    % Pitch Slider and Text
    uicontrol(control_panel, 'Style', 'text', 'String', 'Pitch:', ...
              'Position', [10 y_offset-gap_height label_width slider_height], 'HorizontalAlignment', 'left');

    slider_pitch = uislider(control_panel, 'Value', pitch_deg_init, ...
                            'Position', [10 + label_width y_offset-gap_height slider_width slider_height], ...
                            'Limits', [pitch_deg_init-22.5 pitch_deg_init+22.5]);

    slider_pitch.ValueChangedFcn = @(s,e) updateFigure(fig);

    setappdata(fig,'slider_pitch',slider_pitch);

    text_pitch = uitextarea(control_panel,'Value', num2str(pitch_deg_init, '%.2f'), ...
                       'Position', [30 + label_width + slider_width y_offset-gap_height value_width slider_height]);

    setappdata(fig, 'text_pitch', text_pitch);

    % Roll Slider and Text
    uicontrol(control_panel, 'Style', 'text', 'String', 'Roll:', ...
              'Position', [10 y_offset-2*gap_height label_width slider_height], 'HorizontalAlignment', 'left');

    slider_roll = uislider(control_panel, 'Value', roll_deg_init, ...
                          'Position', [10 + label_width y_offset-2*gap_height slider_width slider_height], ...
                          'Limits',[roll_deg_init-22.5 roll_deg_init+22.5]);

    slider_roll.ValueChangedFcn = @(s,e) updateFigure(fig);

    setappdata(fig,'slider_roll',slider_roll);
                       
    text_roll = uitextarea(control_panel, 'Value', num2str(roll_deg_init, '%.2f'), ...
                       'Position', [30 + label_width + slider_width y_offset-2*gap_height value_width slider_height]);

    setappdata(fig, 'text_roll', text_roll);

    setappdata(fig, 'euler_deg_init', [yaw_deg_init, pitch_deg_init, roll_deg_init]);
    
     %% Set Up Sliders for Positions
    uicontrol(control_panel, 'Style', 'text', 'String', 'X:', ...
              'Position', [10 y_offset-3*gap_height label_width slider_height], 'HorizontalAlignment', 'left');

    slider_x = uislider(control_panel, 'Value', initial_pos(1), ...
                         'Position', [10 + label_width y_offset-3*gap_height slider_width slider_height], ...
                         'Limits', [initial_pos(1)-20 initial_pos(1)+20]);

    slider_x.ValueChangedFcn = @(s,e) updateFigure(fig);

    setappdata(fig,'slider_x',slider_x);

    text_x = uitextarea(control_panel,'Value', num2str(initial_pos(1), '%.2f'), ...
                       'Position', [30 + label_width + slider_width y_offset-3*gap_height value_width slider_height]);

    setappdata(fig, 'text_x', text_x);

    uicontrol(control_panel, 'Style', 'text', 'String', 'Y:', ...
              'Position', [10 y_offset-4*gap_height label_width slider_height], 'HorizontalAlignment', 'left');

    slider_y = uislider(control_panel, 'Value', initial_pos(2), ...
                         'Position', [10 + label_width y_offset-4*gap_height slider_width slider_height], ...
                         'Limits', [initial_pos(2)-20 initial_pos(2)+20]);

    slider_y.ValueChangedFcn = @(s,e) updateFigure(fig);

    setappdata(fig,'slider_y',slider_y);

    text_y = uitextarea(control_panel,'Value', num2str(initial_pos(2), '%.2f'), ...
                       'Position', [30 + label_width + slider_width y_offset-4*gap_height value_width slider_height]);

    setappdata(fig, 'text_y', text_y);
    
    uicontrol(control_panel, 'Style', 'text', 'String', 'Z:', ...
              'Position', [10 y_offset-5*gap_height label_width slider_height], 'HorizontalAlignment', 'left');

    slider_z = uislider(control_panel, 'Value', initial_pos(3), ...
                         'Position', [10 + label_width y_offset-5*gap_height slider_width slider_height], ...
                         'Limits', [initial_pos(3)-20 initial_pos(3)+20]);

    slider_z.ValueChangedFcn = @(s,e) updateFigure(fig);

    setappdata(fig,'slider_z',slider_z);

    text_z = uitextarea(control_panel,'Value', num2str(initial_pos(3), '%.2f'), ...
                       'Position', [30 + label_width + slider_width y_offset-5*gap_height value_width slider_height]);

    setappdata(fig, 'text_z', text_z);

    %% Set Up same stuff but for new set of points
    if ~isempty(points)
        control_panel2 = uipanel(fig, 'Title', 'Point Adjustment Controls', ...
                        'Units', 'pixels', 'Position', [10 fig_pos(4)-gui_height gui_width gui_height], ...
                        'FontSize', 12);

        % Yaw Slider and Text
        uicontrol(control_panel2, 'Style', 'text', 'String', 'dxTh:', ...
                  'Position', [10 y_offset label_width slider_height], 'HorizontalAlignment', 'left');
    
        pslider_yaw = uislider(control_panel2, 'Value', 0, ...
                             'Position', [10 + label_width y_offset slider_width slider_height], ...
                             'Limits', [0 90]);
        pslider_yaw.ValueChangedFcn = @(s,e) updateFigure(fig);
    
        setappdata(fig,'pslider_yaw',pslider_yaw);
    
        ptext_yaw = uitextarea(control_panel2,'Value', num2str(0), ...
                           'Position', [30 + label_width + slider_width y_offset value_width slider_height]);
    
        setappdata(fig, 'ptext_yaw', ptext_yaw);
    
        % Pitch Slider and Text
        uicontrol(control_panel2, 'Style', 'text', 'String', 'dyTh:', ...
                  'Position', [10 y_offset-1*gap_height label_width slider_height], 'HorizontalAlignment', 'left');
    
        pslider_pitch = uislider(control_panel2, 'Value', 0, ...
                                'Position', [10 + label_width y_offset-1*gap_height slider_width slider_height], ...
                                'Limits', [0 90]);
    
        pslider_pitch.ValueChangedFcn = @(s,e) updateFigure(fig);
    
        setappdata(fig,'pslider_pitch',pslider_pitch);
    
        ptext_pitch = uitextarea(control_panel2,'Value', num2str(0), ...
                           'Position', [30 + label_width + slider_width y_offset-1*gap_height value_width slider_height]);
    
        setappdata(fig, 'ptext_pitch', ptext_pitch);
    
        % Roll Slider and Text
        uicontrol(control_panel2, 'Style', 'text', 'String', 'dzTh:', ...
                  'Position', [10 y_offset-2*gap_height label_width slider_height], 'HorizontalAlignment', 'left');
    
        pslider_roll = uislider(control_panel2, 'Value', roll_deg_init, ...
                              'Position', [10 + label_width y_offset-2*gap_height slider_width slider_height], ...
                              'Limits',[0 90]);
    
        pslider_roll.ValueChangedFcn = @(s,e) updateFigure(fig);
    
        setappdata(fig,'pslider_roll',pslider_roll);
                           
        ptext_roll = uitextarea(control_panel2, 'Value', num2str(0), ...
                           'Position', [30 + label_width + slider_width y_offset-2*gap_height value_width slider_height]);
    
        setappdata(fig, 'ptext_roll', ptext_roll);

        % Point Positional Sliders
        uicontrol(control_panel2, 'Style', 'text', 'String', 'dx:', ...
                  'Position', [10 y_offset-3*gap_height label_width slider_height], 'HorizontalAlignment', 'left');
    
        pslider_x = uislider(control_panel2, 'Value', 0, ...
                             'Position', [10 + label_width y_offset-3*gap_height slider_width slider_height], ...
                             'Limits', [-100 100]);
    
        pslider_x.ValueChangedFcn = @(s,e) updateFigure(fig);
    
        setappdata(fig,'pslider_x',pslider_x);
    
        ptext_x = uitextarea(control_panel2,'Value', num2str(0), ...
                           'Position', [30 + label_width + slider_width y_offset-3*gap_height value_width slider_height]);
    
        setappdata(fig, 'ptext_x', ptext_x);
    
        uicontrol(control_panel2, 'Style', 'text', 'String', 'dy:', ...
                  'Position', [10 y_offset-4*gap_height label_width slider_height], 'HorizontalAlignment', 'left');
    
        pslider_y = uislider(control_panel2, 'Value', 0, ...
                             'Position', [10 + label_width y_offset-4*gap_height slider_width slider_height], ...
                             'Limits', [-100 100]);
    
        slider_y.ValueChangedFcn = @(s,e) updateFigure(fig);
    
        setappdata(fig,'pslider_y',pslider_y);
    
        ptext_y = uitextarea(control_panel2,'Value', num2str(0), ...
                           'Position', [30 + label_width + slider_width y_offset-4*gap_height value_width slider_height]);
    
        setappdata(fig, 'ptext_y', ptext_y);
        
        uicontrol(control_panel2, 'Style', 'text', 'String', 'dz:', ...
                  'Position', [10 y_offset-5*gap_height label_width slider_height], 'HorizontalAlignment', 'left');
    
        pslider_z = uislider(control_panel2, 'Value', 0, ...
                             'Position', [10 + label_width y_offset-5*gap_height slider_width slider_height], ...
                             'Limits', [-100 100]);
    
        pslider_z.ValueChangedFcn = @(s,e) updateFigure(fig);
    
        setappdata(fig,'pslider_z',pslider_z);
    
        ptext_z = uitextarea(control_panel2,'Value', num2str(0), ...
                           'Position', [30 + label_width + slider_width y_offset-5*gap_height value_width slider_height]);
    
        setappdata(fig, 'ptext_z', ptext_z);

        prot_buttons = uibuttongroup(control_panel2, ...
        'Title','Adjust with a / d', ...
        'Units','pixels', ...
        'Position',[10 60 130 110]);

        prbYaw   = uiradiobutton(prot_buttons,'Text','dxTh',  'Position',[10 70 80 20],'Value',true);
        prbPitch = uiradiobutton(prot_buttons,'Text','dyTh','Position',[10 45 80 20]);
        prbRoll  = uiradiobutton(prot_buttons,'Text','dzTh', 'Position',[10 20 80 20]);

        setappdata(fig,'prot_buttons',prot_buttons);
        fig.KeyPressFcn = @(src,event) keyPressRouter(src,event);
    
        pmov_buttons = uibuttongroup(control_panel2, ...
        'Title','Adjust with w / s', ...
        'Units','pixels', ...
        'Position',[140 60 130 110]);
    
        prbX   = uiradiobutton(pmov_buttons,'Text','dx',  'Position',[10 70 80 20],'Value',true);
        prbY = uiradiobutton(pmov_buttons,'Text','dy','Position',[10 45 80 20]);
        prbZ  = uiradiobutton(pmov_buttons,'Text','dz', 'Position',[10 20 80 20]);
        
        setappdata(fig,'pmov_buttons',pmov_buttons);
        fig.KeyPressFcn = @(src,event) keyPressRouter(src,event);

        chkHide = uicheckbox(control_panel, ...
          'Text','Hide pts', ...
          'Position',[220 15 70 20], ...   % (x,y,w,h) – tweak as you like
          'Value',false, ...               % start unchecked
          'ValueChangedFcn',@(cb,evt) togglePoints(fig,cb));

        setappdata(fig,'chkHidePts',chkHide);

    end 
    %%
    rot_buttons = uibuttongroup(control_panel, ...
        'Title','Adjust with ← / →', ...
        'Units','pixels', ...
        'Position',[10 60 130 110]);

    rbYaw   = uiradiobutton(rot_buttons,'Text','Yaw',  'Position',[10 70 80 20],'Value',true);
    rbPitch = uiradiobutton(rot_buttons,'Text','Pitch','Position',[10 45 80 20]);
    rbRoll  = uiradiobutton(rot_buttons,'Text','Roll', 'Position',[10 20 80 20]);
    
    setappdata(fig,'rot_buttons',rot_buttons);
    fig.KeyPressFcn = @(src,event) keyPressRouter(src,event);

    mov_buttons = uibuttongroup(control_panel, ...
    'Title','Adjust with ꜛ / ꜜ', ...
    'Units','pixels', ...
    'Position',[140 60 130 110]);

    rbX   = uiradiobutton(mov_buttons,'Text','X',  'Position',[10 70 80 20],'Value',true);
    rbY = uiradiobutton(mov_buttons,'Text','Y','Position',[10 45 80 20]);
    rbZ  = uiradiobutton(mov_buttons,'Text','Z', 'Position',[10 20 80 20]);
    
    setappdata(fig,'mov_buttons',mov_buttons);
    fig.KeyPressFcn = @(src,event) keyPressRouter(src,event);
    
    if ~isempty(second_image)
        switchBtn = uibutton(control_panel, ...
          'Text','Switch Image', ...
          'Position',[ 110 10 100 30 ], ...  
          'ButtonPushedFcn',@(btn,evt) switchBackground(fig));
    end

    doneBtn = uibutton(control_panel, ...
          'Text','Done', ...
          'Position',[ 10 10 100 30 ], ...  
          'ButtonPushedFcn',@(btn,evt) doneCallback(fig));

    uiwait(fig);
    proj_user_points = getappdata(fig, 'manual_world');
    P_final = getappdata(fig, 'P_final');
    delete(fig)
   
end

function updateFigure(fig)
    K = getappdata(fig, 'K');
    pointsHom = getappdata(fig, 'PointsHom');
    vertsHom = getappdata(fig, 'VertsHom');
    image = getappdata(fig, 'image');
    ax = getappdata(fig, 'ax');
    hPatch = getappdata(fig,'hPatch');
    hScatter = getappdata(fig,'hScatter');
    extra_faces = getappdata(fig,'extra_faces');
    initial_pos = getappdata(fig, 'initial_pos');
    initial_targ = getappdata(fig, 'initial_targ');
    q_cw_init = getappdata(fig, 'q_cw_init');
    q_wc_init = getappdata(fig, 'q_wc_init');
    euler_deg_init = getappdata(fig, 'euler_deg_init');
    
    euler_yaw_deg = get(getappdata(fig, 'slider_yaw'), 'Value');
    euler_pitch_deg = get(getappdata(fig, 'slider_pitch'), 'Value');
    euler_roll_deg = get(getappdata(fig, 'slider_roll'), 'Value');
    x_val = get(getappdata(fig,'slider_x'),'Value');               
    y_val = get(getappdata(fig,'slider_y'),'Value');               
    z_val = get(getappdata(fig,'slider_z'),'Value');                
    camPos = [x_val; y_val; z_val];

    % Do everything for the points
    pDxTh = get(getappdata(fig, 'pslider_yaw'), 'Value');
    pDyTh = get(getappdata(fig, 'pslider_pitch'), 'Value');
    pDzTh = get(getappdata(fig, 'pslider_roll'), 'Value');
    pDx = get(getappdata(fig,'pslider_x'),'Value');      
    pDy = get(getappdata(fig,'pslider_y'),'Value'); 
    pDz = get(getappdata(fig,'pslider_z'),'Value');

    qp_yaw   = quaternion([pDxTh 0 0],'eulerd','YXZ','frame');
    qp_pitch = quaternion([0 pDyTh 0],'eulerd','YXZ','frame');
    qp_roll  = quaternion([0 0 pDzTh],'eulerd','YXZ','frame');
    
    qp_total = qp_roll * qp_pitch * qp_yaw;
    Rp       = quat2rotm(compact(qp_total));       
    
    tp       = [pDx; pDy; pDz];               
    
    ptsMoved = Rp * pointsHom(1:3,:) + tp;       
    pointsHom = [ptsMoved; ones(1,size(ptsMoved,2))];
    
    % Update Text Displays
    set(getappdata(fig, 'text_yaw'), 'Value', num2str(euler_yaw_deg, '%.2f'));
    set(getappdata(fig, 'text_pitch'), 'Value', num2str(euler_pitch_deg, '%.2f'));
    set(getappdata(fig, 'text_roll'), 'Value', num2str(euler_roll_deg, '%.2f'));                                           
    set(getappdata(fig,'text_x'),'Value',num2str(x_val,'%.2f'));    
    set(getappdata(fig,'text_y'),'Value',num2str(y_val,'%.2f'));   
    set(getappdata(fig,'text_z'),'Value',num2str(z_val,'%.2f'));    

    set(getappdata(fig, 'ptext_yaw'), 'Value', num2str(pDxTh, '%.2f'));
    set(getappdata(fig, 'ptext_pitch'), 'Value', num2str(pDyTh, '%.2f'));
    set(getappdata(fig, 'ptext_roll'), 'Value', num2str(pDzTh, '%.2f'));                                           
    set(getappdata(fig,'ptext_x'),'Value',num2str(pDx,'%.2f'));    
    set(getappdata(fig,'ptext_y'),'Value',num2str(pDy,'%.2f'));   
    set(getappdata(fig,'ptext_z'),'Value',num2str(pDz,'%.2f'));    
                               
    % Calculate New Rotation Matrix with quaternions
    delta_angles_deg = [euler_yaw_deg, euler_pitch_deg, euler_roll_deg]-euler_deg_init;
    q_yaw_delta   = quaternion([delta_angles_deg(1), 0, 0], 'eulerd', 'YXZ', 'frame');
    q_pitch_delta = quaternion([0, delta_angles_deg(2), 0], 'eulerd', 'YXZ', 'frame');
    q_roll_delta  = quaternion([0, 0, delta_angles_deg(3)], 'eulerd', 'YXZ', 'frame');
    
    q_total = q_roll_delta * q_pitch_delta * q_yaw_delta * q_wc_init;
    R_cw = quat2rotm(compact(q_total));

    % Create New Projection Matrix
    t = -R_cw*camPos;
    L = [R_cw, t];
    P = K*L;
    
    % Save Stuff for Later
    setappdata(fig,'R_cw_current',R_cw);     
    setappdata(fig,'camPos_current',camPos);
    setappdata(fig,'P_current',P);

    % Reproject Vertices
    proj_verts = P*vertsHom;
    proj_points = P*pointsHom;

    % Divide by Z-axis (distance)
    proj_verts_x = proj_verts(1,:)./proj_verts(3,:);
    proj_verts_y = proj_verts(2,:)./proj_verts(3,:);    

    proj_points_x = proj_points(1,:)./proj_points(3,:);
    proj_points_y = proj_points(2,:)./proj_points(3,:);

    set(hPatch,'Vertices', [proj_verts_x', proj_verts_y', zeros(size(proj_verts_x'))]);
    if ~isempty(extra_faces)
        set(hScatter,'Vertices', [proj_points_x', proj_points_y', zeros(size(proj_points_x'))]);
    else
        set(hScatter, 'Xdata', proj_points_x, 'YData', proj_points_y);
    end
    drawnow;
end

function keyPressRouter(fig,event)
    switch event.Key
        case {'a','d'}
            a_d_Callback(fig, event);

        case {'w','s'}
             w_s_callback(fig, event);

        case {'leftarrow','rightarrow'}
            arrowKeyCallback(fig,event);

        case {'uparrow','downarrow'}
            arrowKeyUpDown(fig,event);
    end
end

function arrowKeyCallback(fig,event)
    % Only care about right and left
    if ~ismember(event.Key,{'leftarrow','rightarrow'}) 
        return;  
    end

    % Step size (degrees) per key‑press
    step = 0.25;                     
    if strcmp(event.Key,'leftarrow')  
        step = -step;  
    end

    % Which axis is selected?
    axis_grp = getappdata(fig,'rot_buttons');
    selected = axis_grp.SelectedObject.Text;  

    switch selected
        case 'Yaw',   slider = getappdata(fig,'slider_yaw');
        case 'Pitch', slider = getappdata(fig,'slider_pitch');
        case 'Roll',  slider = getappdata(fig,'slider_roll');
    end

    % Nudge Slider Within Limits
    newVal = min(max(slider.Value + step, slider.Limits(1)), slider.Limits(2));
    slider.Value = newVal;

    updateFigure(fig);
end

function a_d_Callback(fig,event)
    % Only care about right and left
    if ~ismember(event.Key,{'a','d'}) 
        return;  
    end

    % Step size (degrees) per key‑press
    step = 0.25;                     
    if strcmp(event.Key,'a')  
        step = -step;  
    end

    % Which axis is selected?
    axis_grp = getappdata(fig,'prot_buttons');
    selected = axis_grp.SelectedObject.Text;  

    switch selected
        case 'dxTh',   slider = getappdata(fig,'pslider_yaw');
        case 'dyTh', slider = getappdata(fig,'pslider_pitch');
        case 'dzTh',  slider = getappdata(fig,'pslider_roll');
    end

    % Nudge Slider Within Limits
    newVal = min(max(slider.Value + step, slider.Limits(1)), slider.Limits(2));
    slider.Value = newVal;

    updateFigure(fig);
end

function arrowKeyUpDown(fig,event)
    % Only care about up and down
    if ~ismember(event.Key,{'uparrow','downarrow'}) 
        return;  
    end

    % Step size (mm) per key‑press
    step = .25;                     
    if strcmp(event.Key,'downarrow')  
        step = -step;  
    end

    % Which axis is selected?
    axis_grp = getappdata(fig,'mov_buttons');
    selected = axis_grp.SelectedObject.Text;  

    switch selected
        case 'X',   slider = getappdata(fig,'slider_x');
        case 'Y', slider = getappdata(fig,'slider_y');
        case 'Z',  slider = getappdata(fig,'slider_z');
    end

    % Nudge Slider Within Limits
    newVal = min(max(slider.Value + step, slider.Limits(1)), slider.Limits(2));
    slider.Value = newVal;
    updateFigure(fig);
end

function w_s_callback(fig,event)
    % Only care about up and down
    if ~ismember(event.Key,{'w','s'}) 
        return;  
    end

    % Step size (mm) per key‑press
    step = .25;                     
    if strcmp(event.Key,'s')  
        step = -step;  
    end

    % Which axis is selected?
    axis_grp = getappdata(fig,'pmov_buttons');
    selected = axis_grp.SelectedObject.Text;  

    switch selected
        case 'dx',   slider = getappdata(fig,'pslider_x');
        case 'dy', slider = getappdata(fig,'pslider_y');
        case 'dz',  slider = getappdata(fig,'pslider_z');
    end

    % Nudge Slider Within Limits
    newVal = min(max(slider.Value + step, slider.Limits(1)), slider.Limits(2));
    slider.Value = newVal;
    updateFigure(fig);
end

function doneCallback(fig)
    R_fix_model = [0 1 0; -1 0 0; 0 0 1];
    move_model = [256 15 60];
    mriData = getappdata(fig, 'mriData');
    % Take the most recent projection matrix
    P = getappdata(fig,'P_current');
    setappdata(fig,'P_final',P);   
    
    worldPts = imagePtsToWorld(fig);
    if mriData
        worldPts = (worldPts-move_model)*R_fix_model';
        setappdata(fig,'manual_world',worldPts);
    else
        setappdata(fig,'manual_world',worldPts);
    end

    uiresume(fig);
end

function switchBackground(fig)
    % Grab necessary sturf
    bg = getappdata(fig,'bgHandle');
    mainImg = getappdata(fig,'main_image');
    altImg = getappdata(fig,'alt_image');
    showingMain = getappdata(fig,'isMainShowing');

    % Nothing to do if the second image was never supplied
    if isempty(altImg)
        return
    end

    % Toggle the CData
    if showingMain
        set(bg,'CData',altImg);
    else
        set(bg,'CData',mainImg);
    end

    % Flip the flag
    setappdata(fig,'isMainShowing',~showingMain);
end

function markPoint(fig,~)
    ax = getappdata(fig,'ax');
    hManual = getappdata(fig,'hManual');
    pts = getappdata(fig,'manual_points');

    % Image‑space coordinate of the click
    cp = get(ax,'CurrentPoint');
    x  = cp(1,1);                         
    y  = cp(1,2);

    if  x < ax.XLim(1) || x > ax.XLim(2) || ...
        y < ax.YLim(1) || y > ax.YLim(2)
        return
    end

    clickType = get(fig,'SelectionType');
    % Left click adds point, right click deletes
    switch clickType
        case 'normal'
            pts = [pts; x y];

        case 'alt'
            if ~isempty(pts)
                [~,idx] = min(vecnorm(pts - [x y],2,2));
                pts(idx,:) = [];
            end
    end
    % Update graphics & stored data
    set(hManual,'XData',pts(:,1),'YData',pts(:,2));
    setappdata(fig,'manual_points',pts);
end

function worldPts = imagePtsToWorld(fig)
    % Function that Projects 2D manual points to 3D
    faces = getappdata(fig,'faces');
    verts_hom = getappdata(fig,'VertsHom');
    verts = verts_hom(1:3,:).';
    v0 = verts(faces(:,1),:);
    v1 = verts(faces(:,2),:);
    v2 = verts(faces(:,3),:);
    
    if size(verts,1) ~= 3
        verts = verts.';
    end

    if nargin<2, planeZ = 0; end   

    K = getappdata(fig,'K');
    R_cw = getappdata(fig,'R_cw_current');
    C = getappdata(fig,'camPos_current');   
    imgPts = getappdata(fig,'manual_points'); 
    if size(C,1) == 3, C = C.';end
    
    if isempty(imgPts)
        worldPts = [];  return
    end
                       
    N = size(imgPts,1);
    worldPts = zeros(N,3);
    
    % Calculate Ray from Camera 
    uv1 = [imgPts.' ; ones(1,size(imgPts,1))];    
    dirs_c = K\uv1;                       
    dirs_w = R_cw.' * dirs_c;                     
    dirs_w = dirs_w ./ vecnorm(dirs_w);            

    % % Calculate Distances from verts to ray
    V = verts - C';                                 
    V_norm2  = sum(V.^2,1);                               
    t = dirs_w.' * V;                             
    d2 = V_norm2 - t.^2;  

    for k = 1:N
        % replicate the single ray so we can test against every triangle at once
        orig = repmat(C, size(faces,1), 1); 
        dir  = repmat(dirs_w(:,k)', size(faces,1), 1);
    
        % intersect – logical F×1,  t – distance,  xcoor – F×3 hit points
        [intersect, t, ~, ~, xcoor] = TriangleRayIntersection( ...
                orig, dir, v0, v1, v2, 'lineType', 'ray');
    
        % keep only forward hits (t>0) and choose the closest one
        hitIdx = find(intersect & t > 0);
        if ~isempty(hitIdx)
            [~, ii] = min(t(hitIdx));
            worldPts(k,:) = xcoor(hitIdx(ii),:);
        end
    end
end

function togglePoints(fig,chk)
    hScatter = getappdata(fig,'hScatter');
    hManual  = getappdata(fig,'hManual');

    if chk.Value          
        set(hScatter,'Visible','off');
        set(hManual ,'Visible','off');
    else              
        set(hScatter,'Visible','on');
        set(hManual ,'Visible','on');
    end
end
