// Created     : 2018-08-20 9:53:1
// DLL         : C:\Program Files (x86)\DHI\2019\bin\x64\pfs2004.dll
// Version     : 17.0.0.12230

[FemEngineSW]
   [DOMAIN]
      Touched = 1
      discretization = 2
      number_of_dimensions = 2
      number_of_meshes = 1
      file_name = |.\Lake_Mesh.mesh|
      type_of_reordering = 1
      number_of_domains = 16
      coordinate_type = 'UTM-32'
      minimum_depth = 0.0
      datum_depth = 0.0
      vertical_mesh_type_overall = 1
      number_of_layers = 11
      z_sigma = 0.0
      vertical_mesh_type = 1
      layer_thickness = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0
      sigma_c = 0.0
      theta = 2.0
      b = 0.0
      number_of_layers_zlevel = 10
      vertical_mesh_type_zlevel = 1
      constant_layer_thickness_zlevel = 0.0
      variable_layer_thickness_zlevel = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
      type_of_bathymetry_adjustment = 1
      minimum_layer_thickness_zlevel = 0.0
      type_of_mesh = 0
      type_of_gauss = 3
      [BOUNDARY_NAMES]
         Touched = 0
         MzSEPfsListItemCount = 0
      EndSect  // BOUNDARY_NAMES

   EndSect  // DOMAIN

   [TIME]
      Touched = 1
      start_time = 2002, 1, 1, 0, 0, 0
      time_step_interval = 120.0
      number_of_time_steps = 450
   EndSect  // TIME

   [MODULE_SELECTION]
      Touched = 0
      mode_of_hydrodynamic_module = 0
      hydrodynamic_features = 1
      fluid_property = 1
      mode_of_spectral_wave_module = 2
      mode_of_transport_module = 0
      mode_of_mud_transport_module = 0
      mode_of_eco_lab_module = 0
      mode_of_sand_transport_module = 0
      mode_of_particle_tracking_module = 0
      mode_of_oil_spill_module = 0
      mode_of_shoreline_module = 0
   EndSect  // MODULE_SELECTION

   [SPECTRAL_WAVE_MODULE]
      mode = 2
      [SPACE]
         number_of_mesh_geometry = 1
      EndSect  // SPACE

      [EQUATION]
         Touched = 1
         formulation = 2
         time_formulation = 2
         JONSWAP_factor_1 = 0.92
         JONSWAP_factor_2 = 0.83
      EndSect  // EQUATION

      [TIME]
         Touched = 0
         start_time_step = 0
         time_step_factor = 1
         time_step_factor_AD = 1
      EndSect  // TIME

      [SPECTRAL]
         Touched = 1
         type_of_frequency_discretization = 2
         number_of_frequencies = 25
         minimum_frequency = 0.055
         frequency_interval = 0.02
         frequency_factor = 1.1
         type_of_directional_discretization = 1
         number_of_directions = 16
         minimum_direction = 0.0
         maximum_direction = 0.0
         separation_of_wind_sea_and_swell = 0
         threshold_frequency = 0.125
         maximum_threshold_frequency = 0.5959088268863615
      EndSect  // SPECTRAL

      [SOLUTION_TECHNIQUE]
         Touched = 1
         error_level = 0
         maximum_number_of_errors = 200
         minimum_period = 0.1
         maximum_period = 25.0
         initial_period = 8.0
         scheme_of_space_discretization_geographical = 1
         scheme_of_space_discretization_direction = 1
         scheme_of_space_discretization_frequency = 1
         method = 2
         number_of_iterations = 10
         tolerance1 = 0.0001
         tolerance2 = 0.001
         relaxation_factor = 0.1
         number_of_levels_in_transport_calc = 32
         number_of_steps_in_source_calc = 1
         maximum_CFL_number = 1.0
         dt_min = 0.01
         dt_max = 120.0
         type_overall = 0
         file_name_overall = |.\convergence_overall.dfs0|
         input_format = 1
         coordinate_type = ''
         input_file_name = ||
         number_of_points = 0
         type_domain = 0
         file_name_domain = |.\convergence_domain.dfsu|
         output_frequency = 5
      EndSect  // SOLUTION_TECHNIQUE

      [DEPTH]
         Touched = 1
         type = 0
         minimum_depth = 0.01
         format = 0
         soft_time_interval = 0.0
         constant_level = 0.0
         file_name = ||
         item_number = 1
         item_name = ''
      EndSect  // DEPTH

      [CURRENT]
         Touched = 1
         type = 0
         type_blocking = 1
         factor_blocking = 0.1
         format = 0
         soft_time_interval = 0.0
         constant_x_velocity = 0.0
         constant_y_velocity = 0.0
         file_name = ||
         item_number_for_x_velocity = 0
         item_number_for_y_velocity = 0
         item_name_for_x_velocity = ''
         item_name_for_y_velocity = ''
      EndSect  // CURRENT

      [WIND]
         Touched = 1
         type = 1
         format = 1
         constant_speed = 15.0
         constant_direction = 270.0
         file_name = |.\Wind.dfs0|
         item_number_for_speed = 1
         item_number_for_direction = 2
         item_name_for_speed = 'Speed'
         item_name_for_direction = 'Direction'
         soft_time_interval = 0.0
         formula = 1
         type_of_drag = 1
         linear_growth_coefficient = 0.0015
         type_of_air_sea_interaction = 1
         background_Charnock_parameter = 0.01
         Charnock_parameter = 0.01
         alpha_drag = 0.00063
         beta_drag = 6.600000000000001e-05
      EndSect  // WIND

      [ICE]
         Touched = 1
         type = 0
         format = 3
         c_cut_off = 0.33
         file_name = ||
         item_number = 1
         item_name = ''
      EndSect  // ICE

      [DIFFRACTION]
         Touched = 1
         type = 0
         minimum_delta = -0.75
         maximum_delta = 3.0
         type_of_smoothing = 1
         smoothing_factor = 1.0
         number_of_smoothing_steps = 1
      EndSect  // DIFFRACTION

      [TRANSFER]
         Touched = 1
         type = 1
         type_triad = 0
         alpha_EB = 0.25
      EndSect  // TRANSFER

      [WAVE_BREAKING]
         Touched = 1
         type = 0
         type_of_gamma = 1
         alpha = 1.0
         gamma_steepness = 1.0
         type_of_effect_on_frequency = 1
         type_of_roller = 0
         roller_propagation_factor = 1.0
         roller_dissipation_factor = 0.15
         roller_density = 1000.0
         [GAMMA]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.55
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // GAMMA

      EndSect  // WAVE_BREAKING

      [BOTTOM_FRICTION]
         Touched = 1
         type = 0
         constant_fc = 0.0
         type_of_effect_on_frequency = 0
         [FRICTION_COEFFICIENT]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.0775
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // FRICTION_COEFFICIENT

         [FRICTION_FACTOR]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.0212
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // FRICTION_FACTOR

         [NIKURADSE_ROUGHNESS]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.04
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // NIKURADSE_ROUGHNESS

         [SAND_GRAIN_SIZE]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.00025
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // SAND_GRAIN_SIZE

      EndSect  // BOTTOM_FRICTION

      [WHITECAPPING]
         Touched = 1
         type = 1
         type_of_spectrum = 3
         mean_frequency_power = -1
         mean_wave_number_power = -1
         [dissipation_cdiss]
            Touched = 1
            type = 1
            format = 0
            constant_value = 4.5
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // dissipation_cdiss

         [dissipation_delta]
            Touched = 1
            type = 1
            format = 0
            constant_value = 0.5
            file_name = ||
            item_number = 1
            item_name = ''
            type_of_soft_start = 2
            soft_time_interval = 0.0
            reference_value = 0.0
            type_of_time_interpolation = 1
         EndSect  // dissipation_delta

      EndSect  // WHITECAPPING

      [STRUCTURES]
         type = 0
         input_format = 1
         coordinate_type = ''
         number_of_structures = 0
         input_file_name = ||
         [LINE_STRUCTURES]
            Touched = 1
            MzSEPfsListItemCount = 0
            output_of_link_data = 0
            file_name_section = 'line_section.xyz'
            number_of_structures = 0
         EndSect  // LINE_STRUCTURES

      EndSect  // STRUCTURES

      [INITIAL_CONDITIONS]
         Touched = 1
         type = 0
         type_additional = 1
         type_of_spectra = 1
         fetch = 40000.0
         max_peak_frequency = 0.4
         max_Phillips_constant = 0.0081
         shape_parameter_sigma_a = 0.07000000000000001
         shape_parameter_sigma_b = 0.09
         peakednes_parameter_gamma = 3.3
         file_name_m = ||
         item_number_m0 = 1
         item_number_m1 = 1
         item_name_m0 = ''
         item_name_m1 = ''
         file_name_A = ||
         item_number_A = 1
         item_name_A = ''
      EndSect  // INITIAL_CONDITIONS

      [BOUNDARY_CONDITIONS]
         Touched = 0
         MzSEPfsListItemCount = 0
         [CODE_1]
         EndSect  // CODE_1

      EndSect  // BOUNDARY_CONDITIONS

      [OUTPUTS]
         Touched = 1
         MzSEPfsListItemCount = 4
         number_of_outputs = 4
         [OUTPUT_1]
            Touched = 1
            include = 1
            title = 'Wave parameters in domain'
            file_name = 'Wave_parameters.dfsu'
            type = 1
            format = 2
            flood_and_dry = 2
            coordinate_type = 'UTM-32'
            zone = 0
            input_file_name = ||
            input_format = 1
            interpolation_type = 1
            use_end_time = 1
            first_time_step = 0
            last_time_step = 450
            time_step_frequency = 10
            number_of_points = 1
            [POINT_1]
               name = 'POINT_1'
               x = 20000.0
               y = 20000.0
            EndSect  // POINT_1

            [LINE]
               npoints = 3
               x_first = 0.0
               y_first = 0.0
               x_last = 40000.0
               y_last = 40000.0
            EndSect  // LINE

            [AREA]
               number_of_points = 4
               [POINT_1]
                  x = -400.0
                  y = -400.0
               EndSect  // POINT_1

               [POINT_2]
                  x = -400.0
                  y = 40400.0
               EndSect  // POINT_2

               [POINT_3]
                  x = 40400.0
                  y = 40400.0
               EndSect  // POINT_3

               [POINT_4]
                  x = 40400.0
                  y = -400.0
               EndSect  // POINT_4

               orientation = 0.0
               x_origo = 0.0
               x_ds = 2105.263157894737
               x_npoints = 20
               y_origo = 0.0
               y_ds = 2105.263157894737
               y_npoints = 20
               z_origo = -15.00000000000001
               z_ds = 2.000000000000001
               z_npoints = 10
            EndSect  // AREA

            [INTEGRAL_WAVE_PARAMETERS]
               Touched = 1
               type_of_spectrum = 1
               minimum_frequency = 0.055
               maximum_frequency = 0.5959088268863617
               separation_of_wind_sea_and_swell = 3
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               hm0_minimum = 0.01
               type_of_h_max = 3
               duration = 10800.0
               distance_above_bed_for_particle_velocity = 0.0
               minimum_direction = 0.0
               maximum_direction = 360.0
               [Total_wave_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 0
                  Peak_wave_period = 1
                  Wave_period_t01 = 0
                  Wave_period_t02 = 1
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 1
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 1
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Total_wave_parameters

               [Wind_sea_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Wind_sea_parameters

               [Swell_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Swell_parameters

            EndSect  // INTEGRAL_WAVE_PARAMETERS

            [INPUT_PARAMETERS]
               Touched = 1
               Surface_elevation = 0
               Water_depth = 0
               Current_velocity_components = 0
               Wind_speed = 0
               Wind_direction = 0
               Ice_concentration = 0
            EndSect  // INPUT_PARAMETERS

            [MODEL_PARAMETERS]
               Touched = 1
               Wind_friction_speed = 0
               Roughness_length = 0
               Drag_coefficient = 0
               Charnock_constant = 0
               Friction_coefficient = 0
               Breaking_parameter_gamma = 0
               Courant_number = 1
               Time_step_factor = 1
               Convergence_angle = 0
               Length = 0
               Area = 0
               Threshold_period = 0
               Roller_area = 0
               Roller_dissipation = 0
               Breaking_index = 0
            EndSect  // MODEL_PARAMETERS

            [SPECTRAL_PARAMETERS]
               Touched = 1
               separation_of_wind_sea_and_swell = 3.0
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               wave_energy = 1
               wave_action = 0
               zeroth_moment_of_wave_action = 0
               first_moment_of_wave_action = 0
               wave_energy_wind_sea = 0
               wave_energy_swell = 0
            EndSect  // SPECTRAL_PARAMETERS

         EndSect  // OUTPUT_1

         [OUTPUT_2]
            Touched = 1
            include = 0
            title = 'Wave parameters along line'
            file_name = 'Wave_line.dfs1'
            type = 1
            format = 1
            flood_and_dry = 2
            coordinate_type = 'UTM-32'
            zone = 0
            input_file_name = ||
            input_format = 1
            interpolation_type = 2
            use_end_time = 1
            first_time_step = 0
            last_time_step = 450
            time_step_frequency = 10
            number_of_points = 1
            [POINT_1]
               name = 'POINT_1'
               x = 20000.0
               y = 20000.0
            EndSect  // POINT_1

            [LINE]
               npoints = 41
               x_first = 0.0
               y_first = 20000.0
               x_last = 40000.0
               y_last = 20000.0
            EndSect  // LINE

            [AREA]
               number_of_points = 4
               [POINT_1]
                  x = -400.0
                  y = -400.0
               EndSect  // POINT_1

               [POINT_2]
                  x = -400.0
                  y = 40400.0
               EndSect  // POINT_2

               [POINT_3]
                  x = 40400.0
                  y = 40400.0
               EndSect  // POINT_3

               [POINT_4]
                  x = 40400.0
                  y = -400.0
               EndSect  // POINT_4

               orientation = 0.0
               x_origo = 0.0
               x_ds = 2105.263157894737
               x_npoints = 20
               y_origo = 0.0
               y_ds = 2105.263157894737
               y_npoints = 20
               z_origo = -15.00000000000001
               z_ds = 2.000000000000001
               z_npoints = 10
            EndSect  // AREA

            [INTEGRAL_WAVE_PARAMETERS]
               Touched = 1
               type_of_spectrum = 1
               minimum_frequency = 0.055
               maximum_frequency = 0.5959088268863617
               separation_of_wind_sea_and_swell = 3
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               hm0_minimum = 0.01
               type_of_h_max = 3
               duration = 10800.0
               distance_above_bed_for_particle_velocity = 0.0
               minimum_direction = 0.0
               maximum_direction = 360.0
               [Total_wave_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 0
                  Peak_wave_period = 1
                  Wave_period_t01 = 0
                  Wave_period_t02 = 1
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 1
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Total_wave_parameters

               [Wind_sea_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Wind_sea_parameters

               [Swell_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Swell_parameters

            EndSect  // INTEGRAL_WAVE_PARAMETERS

            [INPUT_PARAMETERS]
               Touched = 1
               Surface_elevation = 0
               Water_depth = 0
               Current_velocity_components = 0
               Wind_speed = 0
               Wind_direction = 0
               Ice_concentration = 0
            EndSect  // INPUT_PARAMETERS

            [MODEL_PARAMETERS]
               Touched = 1
               Wind_friction_speed = 0
               Roughness_length = 0
               Drag_coefficient = 0
               Charnock_constant = 0
               Friction_coefficient = 0
               Breaking_parameter_gamma = 0
               Courant_number = 0
               Time_step_factor = 0
               Convergence_angle = 0
               Length = 0
               Area = 0
               Threshold_period = 0
               Roller_area = 0
               Roller_dissipation = 0
               Breaking_index = 0
            EndSect  // MODEL_PARAMETERS

            [SPECTRAL_PARAMETERS]
               Touched = 1
               separation_of_wind_sea_and_swell = 3.0
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               wave_energy = 1
               wave_action = 0
               zeroth_moment_of_wave_action = 0
               first_moment_of_wave_action = 0
               wave_energy_wind_sea = 0
               wave_energy_swell = 0
            EndSect  // SPECTRAL_PARAMETERS

         EndSect  // OUTPUT_2

         [OUTPUT_3]
            Touched = 1
            include = 1
            title = 'Wave parameters  in a point'
            file_name = 'Waves_x20km_y20km.dfs0'
            type = 1
            format = 0
            flood_and_dry = 2
            coordinate_type = 'UTM-32'
            zone = 0
            input_file_name = ||
            input_format = 1
            interpolation_type = 2
            use_end_time = 1
            first_time_step = 0
            last_time_step = 450
            time_step_frequency = 1
            number_of_points = 1
            [POINT_1]
               name = 'POINT_1'
               x = 38000.0
               y = 20000.0
            EndSect  // POINT_1

            [LINE]
               npoints = 3
               x_first = 0.0
               y_first = 0.0
               x_last = 40000.0
               y_last = 40000.0
            EndSect  // LINE

            [AREA]
               number_of_points = 4
               [POINT_1]
                  x = -400.0
                  y = -400.0
               EndSect  // POINT_1

               [POINT_2]
                  x = -400.0
                  y = 40400.0
               EndSect  // POINT_2

               [POINT_3]
                  x = 40400.0
                  y = 40400.0
               EndSect  // POINT_3

               [POINT_4]
                  x = 40400.0
                  y = -400.0
               EndSect  // POINT_4

               orientation = 0.0
               x_origo = 0.0
               x_ds = 2105.263157894737
               x_npoints = 20
               y_origo = 0.0
               y_ds = 2105.263157894737
               y_npoints = 20
               z_origo = -15.00000000000001
               z_ds = 2.000000000000001
               z_npoints = 10
            EndSect  // AREA

            [INTEGRAL_WAVE_PARAMETERS]
               Touched = 1
               type_of_spectrum = 1
               minimum_frequency = 0.055
               maximum_frequency = 0.5959088268863617
               separation_of_wind_sea_and_swell = 3
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               hm0_minimum = 0.01
               type_of_h_max = 3
               duration = 10800.0
               distance_above_bed_for_particle_velocity = 0.0
               minimum_direction = 0.0
               maximum_direction = 360.0
               [Total_wave_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 0
                  Peak_wave_period = 1
                  Wave_period_t01 = 0
                  Wave_period_t02 = 1
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 1
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Total_wave_parameters

               [Wind_sea_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Wind_sea_parameters

               [Swell_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Swell_parameters

            EndSect  // INTEGRAL_WAVE_PARAMETERS

            [INPUT_PARAMETERS]
               Touched = 1
               Surface_elevation = 0
               Water_depth = 0
               Current_velocity_components = 0
               Wind_speed = 0
               Wind_direction = 0
               Ice_concentration = 0
            EndSect  // INPUT_PARAMETERS

            [MODEL_PARAMETERS]
               Touched = 1
               Wind_friction_speed = 0
               Roughness_length = 0
               Drag_coefficient = 0
               Charnock_constant = 0
               Friction_coefficient = 0
               Breaking_parameter_gamma = 0
               Courant_number = 0
               Time_step_factor = 0
               Convergence_angle = 0
               Length = 0
               Area = 0
               Threshold_period = 0
               Roller_area = 0
               Roller_dissipation = 0
               Breaking_index = 0
            EndSect  // MODEL_PARAMETERS

            [SPECTRAL_PARAMETERS]
               Touched = 1
               separation_of_wind_sea_and_swell = 3.0
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               wave_energy = 1
               wave_action = 0
               zeroth_moment_of_wave_action = 0
               first_moment_of_wave_action = 0
               wave_energy_wind_sea = 0
               wave_energy_swell = 0
            EndSect  // SPECTRAL_PARAMETERS

         EndSect  // OUTPUT_3

         [OUTPUT_4]
            Touched = 1
            include = 1
            title = 'Spectrum in a point'
            file_name = 'spectrum_x20km_y20km.dfsu'
            type = 4
            format = 0
            flood_and_dry = 2
            coordinate_type = 'UTM-32'
            zone = 0
            input_file_name = ||
            input_format = 1
            interpolation_type = 2
            use_end_time = 1
            first_time_step = 0
            last_time_step = 450
            time_step_frequency = 10
            number_of_points = 1
            [POINT_1]
               name = 'POINT_1'
               x = 38000.0
               y = 20000.0
            EndSect  // POINT_1

            [LINE]
               npoints = 3
               x_first = 0.0
               y_first = 0.0
               x_last = 40000.0
               y_last = 40000.0
            EndSect  // LINE

            [AREA]
               number_of_points = 4
               [POINT_1]
                  x = -400.0
                  y = -400.0
               EndSect  // POINT_1

               [POINT_2]
                  x = -400.0
                  y = 40400.0
               EndSect  // POINT_2

               [POINT_3]
                  x = 40400.0
                  y = 40400.0
               EndSect  // POINT_3

               [POINT_4]
                  x = 40400.0
                  y = -400.0
               EndSect  // POINT_4

               orientation = 0.0
               x_origo = 0.0
               x_ds = 2105.263157894737
               x_npoints = 20
               y_origo = 0.0
               y_ds = 2105.263157894737
               y_npoints = 20
               z_origo = -15.00000000000001
               z_ds = 2.000000000000001
               z_npoints = 10
            EndSect  // AREA

            [INTEGRAL_WAVE_PARAMETERS]
               Touched = 1
               type_of_spectrum = 1
               minimum_frequency = 0.054321
               maximum_frequency = 0.09876543209999999
               separation_of_wind_sea_and_swell = 3
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               hm0_minimum = 0.01
               type_of_h_max = 3
               duration = 10800.0
               distance_above_bed_for_particle_velocity = 0.0
               minimum_direction = 0.0
               maximum_direction = 360.0
               [Total_wave_parameters]
                  Significant_wave_height = 1
                  Maximum_wave_height = 1
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 1
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 1
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 1
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Total_wave_parameters

               [Wind_sea_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Wind_sea_parameters

               [Swell_parameters]
                  Significant_wave_height = 0
                  Maximum_wave_height = 0
                  Peak_wave_period = 0
                  Wave_period_t01 = 0
                  Wave_period_t02 = 0
                  Wave_period_tm10 = 0
                  Peak_wave_direction = 0
                  Mean_wave_direction = 0
                  Directional_standard_deviation = 0
                  Wave_velocity_components = 0
                  Radiation_stresses = 0
                  Particle_velocities = 0
                  Wave_power = 0
               EndSect  // Swell_parameters

            EndSect  // INTEGRAL_WAVE_PARAMETERS

            [INPUT_PARAMETERS]
               Touched = 1
               Surface_elevation = 0
               Water_depth = 0
               Current_velocity_components = 0
               Wind_speed = 0
               Wind_direction = 0
               Ice_concentration = 0
            EndSect  // INPUT_PARAMETERS

            [MODEL_PARAMETERS]
               Touched = 1
               Wind_friction_speed = 0
               Roughness_length = 0
               Drag_coefficient = 0
               Charnock_constant = 0
               Friction_coefficient = 0
               Breaking_parameter_gamma = 0
               Courant_number = 0
               Time_step_factor = 0
               Convergence_angle = 0
               Length = 0
               Area = 0
               Threshold_period = 0
               Roller_area = 0
               Roller_dissipation = 0
               Breaking_index = 0
            EndSect  // MODEL_PARAMETERS

            [SPECTRAL_PARAMETERS]
               Touched = 1
               separation_of_wind_sea_and_swell = 3.0
               threshold_frequency = 0.125
               maximum_threshold_frequency = 0.125
               wave_energy = 1
               wave_action = 0
               zeroth_moment_of_wave_action = 0
               first_moment_of_wave_action = 0
               wave_energy_wind_sea = 0
               wave_energy_swell = 0
            EndSect  // SPECTRAL_PARAMETERS

         EndSect  // OUTPUT_4

      EndSect  // OUTPUTS

   EndSect  // SPECTRAL_WAVE_MODULE

EndSect  // FemEngineSW

