// Created     : 2022-09-12 14:11:25
// DLL         : C:\Program Files (x86)\DHI\MIKE Zero\2021\bin\x64\pfs2004.dll
// Version     : 19.0.0.14309

[FemEngineSW]
	[DOMAIN]
		touched = 1
		discretization = 2
		number_of_dimensions = 2
		number_of_meshes = 1
		file_name = |..\input\SW_local_DWF_MSL_02NAFE_coarsev2.mesh|
		type_of_reordering = 1
		number_of_domains = 16
		coordinate_type = 'LONG/LAT'
		minimum_depth = -2.0
		datum_depth = 0.0
		vertical_mesh_type_overall = 1
		number_of_layers = 10
		z_sigma = -1378.329124307562
		vertical_mesh_type = 1
		layer_thickness = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
		sigma_c = 0.1
		theta = 2.0
		b = 0.0
		number_of_layers_zlevel = 10
		vertical_mesh_type_zlevel = 1
		constant_layer_thickness_zlevel = 137.8329124307562
		variable_layer_thickness_zlevel = 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562, 137.8329124307562
		type_of_bathymetry_adjustment = 2
		minimum_layer_thickness_zlevel = 1.378329124307562
		type_of_mesh = 0
		type_of_gauss = 3
		[BOUNDARY_NAMES]
			touched = 0
			mzsepfslistitemcount = 3
			[CODE_3]
				touched = 0
				name = 'North'
			EndSect  // CODE_3

			[CODE_5]
				touched = 0
				name = 'South'
			EndSect  // CODE_5

			[CODE_1]
				touched = 0
				name = 'Landy'
			EndSect  // CODE_1

		EndSect  // BOUNDARY_NAMES

	EndSect  // DOMAIN

	[TIME]
		touched = 1
		start_time = 2017, 11, 1, 0, 0, 0
		time_step_interval = 600.0
		number_of_time_steps = 4320
	EndSect  // TIME

	[MODULE_SELECTION]
		touched = 1
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
			touched = 1
			formulation = 2
			time_formulation = 2
			jonswap_factor_1 = 0.92
			jonswap_factor_2 = 0.83
		EndSect  // EQUATION

		[TIME]
			touched = 1
			start_time_step = 0
			time_step_factor = 1
			time_step_factor_ad = 1
		EndSect  // TIME

		[SPECTRAL]
			touched = 1
			type_of_frequency_discretization = 2
			number_of_frequencies = 25
			minimum_frequency = 0.055
			frequency_interval = 0.02
			frequency_factor = 1.1
			type_of_directional_discretization = 1
			number_of_directions = 16
			minimum_direction = 0.0
			maximum_direction = 180.0
			separation_of_wind_sea_and_swell = 0
			threshold_frequency = 0.125
			maximum_threshold_frequency = 0.5959088268863615
		EndSect  // SPECTRAL

		[SOLUTION_TECHNIQUE]
			touched = 1
			error_level = 0
			maximum_number_of_errors = 200
			minimum_period = 0.1
			maximum_period = 25.0
			initial_period = 8.0
			scheme_of_space_discretization_geographical = 1
			scheme_of_space_discretization_direction = 1
			scheme_of_space_discretization_frequency = 1
			method = 1
			number_of_iterations = 500
			tolerance1 = 1e-06
			tolerance2 = 0.001
			relaxation_factor = 0.1
			number_of_levels_in_transport_calc = 32
			number_of_steps_in_source_calc = 1
			maximum_cfl_number = 1.0
			dt_min = 0.01
			dt_max = 600.0
			type_overall = 0
			file_name_overall = 'convergence_overall.dfs0'
			input_format = 1
			coordinate_type = ''
			input_file_name = None
			number_of_points = 0
			type_domain = 0
			file_name_domain = 'convergence_domain.dfsu'
			output_frequency = 5
		EndSect  // SOLUTION_TECHNIQUE

		[DEPTH]
			touched = 1
			type = 0
			minimum_depth = 0.01
			format = 3
			soft_time_interval = 0.0
			constant_level = 0.0
			file_name = |..\input\areaHD.dfsu|
			item_number = 1
			item_name = 'Surface elevation'
		EndSect  // DEPTH

		[CURRENT]
			touched = 1
			type = 0
			type_blocking = 1
			factor_blocking = 0.1
			format = 3
			soft_time_interval = 0.0
			constant_x_velocity = 0.0
			constant_y_velocity = 0.0
			file_name = |..\input\areaHD.dfsu|
			item_number_for_x_velocity = 2
			item_number_for_y_velocity = 3
			item_name_for_x_velocity = 'Current velocity, U'
			item_name_for_y_velocity = 'Current velocity, V'
		EndSect  // CURRENT

		[WIND]
			touched = 1
			type = 1
			format = 1
			constant_speed = 0.0
			constant_direction = 0.0
			file_name = |..\input\wind.dfs0|
			item_number_for_speed = 1
			item_number_for_direction = 2
			item_name_for_speed = 'Wind speed'
			item_name_for_direction = 'Wind direction'
			soft_time_interval = 0.0
			formula = 1
			type_of_drag = 1
			linear_growth_coefficient = 0.0015
			type_of_air_sea_interaction = 0
			background_charnock_parameter = '$$CHARNOCK$$'
			charnock_parameter = 0.0116769
			alpha_drag = 0.00063
			beta_drag = 6.6e-05
			[CORRECTION_OF_FRICTION_VELOCITY]
				type = 0
				cap_value = 0.06
			EndSect  // CORRECTION_OF_FRICTION_VELOCITY

		EndSect  // WIND

		[ICE]
			touched = 1
			type = 0
			format = 3
			c_cut_off = 0.33
			file_name = None
			item_number = 1
			item_name = ''
		EndSect  // ICE

		[DIFFRACTION]
			touched = 1
			type = 0
			minimum_delta = -0.75
			maximum_delta = 3.0
			type_of_smoothing = 1
			smoothing_factor = 1.0
			number_of_smoothing_steps = 1
		EndSect  // DIFFRACTION

		[TRANSFER]
			touched = 1
			type = 1
			type_triad = 0
			alpha_eb = 0.25
		EndSect  // TRANSFER

		[WAVE_BREAKING]
			touched = 1
			type = 1
			type_of_gamma = 1
			alpha = 1.0
			gamma_steepness = 1.0
			type_of_effect_on_frequency = 0
			type_of_roller = 0
			roller_propagation_factor = 1.0
			roller_dissipation_factor = 0.15
			roller_density = 1000.0
			[GAMMA]
				touched = 1
				type = 1
				format = 0
				constant_value = 0.8
				file_name = None
				item_number = 1
				item_name = ''
				type_of_soft_start = 2
				soft_time_interval = 0.0
				reference_value = 0.0
				type_of_time_interpolation = 1
			EndSect  // GAMMA

		EndSect  // WAVE_BREAKING

		[BOTTOM_FRICTION]
			touched = 1
			type = 3
			constant_fc = 0.0
			type_of_effect_on_frequency = 1
			[FRICTION_COEFFICIENT]
				touched = 1
				type = 1
				format = 0
				constant_value = 0.0077
				file_name = None
				item_number = 1
				item_name = ''
				type_of_soft_start = 2
				soft_time_interval = 0.0
				reference_value = 0.0
				type_of_time_interpolation = 1
			EndSect  // FRICTION_COEFFICIENT

			[FRICTION_FACTOR]
				touched = 1
				type = 1
				format = 0
				constant_value = 0.0212
				file_name = None
				item_number = 1
				item_name = ''
				type_of_soft_start = 2
				soft_time_interval = 0.0
				reference_value = 0.0
				type_of_time_interpolation = 1
			EndSect  // FRICTION_FACTOR

			[NIKURADSE_ROUGHNESS]
				touched = 1
				type = 1
				format = 0
				constant_value = '$$ROUGHNESS$$'
				file_name = |..\input\HKZN_mesh_v2_final_BF_map.dfsu|
				item_number = 1
				item_name = ''
				type_of_soft_start = 2
				soft_time_interval = 0.0
				reference_value = 0.0
				type_of_time_interpolation = 1
			EndSect  // NIKURADSE_ROUGHNESS

			[SAND_GRAIN_SIZE]
				touched = 1
				type = 1
				format = 0
				constant_value = 0.00025
				file_name = None
				item_number = 1
				item_name = ''
				type_of_soft_start = 2
				soft_time_interval = 0.0
				reference_value = 0.0
				type_of_time_interpolation = 1
			EndSect  // SAND_GRAIN_SIZE

		EndSect  // BOTTOM_FRICTION

		[WHITECAPPING]
			touched = 1
			type = 1
			type_of_spectrum = 3
			mean_frequency_power = 1
			mean_wave_number_power = 1
			[dissipation_cdiss]
				touched = 1
				type = 1
				format = 0
				constant_value = 1.1033
				file_name = None
				item_number = 1
				item_name = ''
				type_of_soft_start = 2
				soft_time_interval = 0.0
				reference_value = 0.0
				type_of_time_interpolation = 1
			EndSect  // dissipation_cdiss

			[dissipation_delta]
				touched = 1
				type = 1
				format = 0
				constant_value = 0.337
				file_name = None
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
			input_file_name = None
			[LINE_STRUCTURES]
				touched = 1
				mzsepfslistitemcount = 0
				output_of_link_data = 0
				file_name_section = 'line_section.xyz'
				number_of_structures = 0
			EndSect  // LINE_STRUCTURES

		EndSect  // STRUCTURES

		[INITIAL_CONDITIONS]
			touched = 1
			type = 0
			type_additional = 1
			type_of_spectra = 1
			fetch = 100000.0
			max_peak_frequency = 0.4
			max_phillips_constant = 0.0081
			shape_parameter_sigma_a = 0.07
			shape_parameter_sigma_b = 0.09
			peakednes_parameter_gamma = 3.3
			file_name_m = None
			item_number_m0 = 1
			item_number_m1 = 1
			item_name_m0 = ''
			item_name_m1 = ''
			file_name_a = None
			item_number_a = 1
			item_name_a = ''
		EndSect  // INITIAL_CONDITIONS

		[BOUNDARY_CONDITIONS]
			touched = 1
			mzsepfslistitemcount = 3
			[CODE_1]
				touched = 1
				type = 1
				format = 0
				constant_values = 1.0, 8.0, 270.0, 5.0, 0.1, 16.0, 270.0, 32.0
				file_name = None
				item_numbers = 1701080931, 12639, 1, 0, 6, 0, 15, 0
				item_names = '', '', '', '', '', '', '', ''
				type_of_soft_start = 1
				soft_time_interval = 0.0
				reference_values = 0.0, 8.0, 270.0, 5.0, 0.0, 16.0, 270.0, 32.0
				type_of_time_interpolation = 1, 1, 1, 1, 1, 1, 1, 1
				type_of_space_interpolation = 1
				code_cyclic = 0
				reflection_coefficient = 1.0
				type_of_frequency_spectrum = 2
				type_of_frequency_normalization = 1
				sigma_a = 0.07
				sigma_b = 0.09
				gamma = 3.3
				type_of_directional_distribution = 1
				type_of_directional_normalization = 1
				type_of_frequency_spectrum_swell = 2
				type_of_frequency_normalization_swell = 1
				sigma_a_swell = 0.07
				sigma_b_swell = 0.09
				gamma_swell = 5.0
				type_of_directional_distribution_swell = 1
				type_of_directional_normalization_swell = 1
			EndSect  // CODE_1

			[CODE_3]
				touched = 1
				type = 4
				format = 3
				constant_values = 1.0, 8.0, 270.0, 5.0, 0.1, 16.0, 270.0, 32.0
				file_name = |..\input\NorthBC_Line.dfs1|
				item_numbers = 1, 2, 3, 4, 1, 1, 1, 1
				item_names = 'Action density', 'Action density', 'Action density', 'Action density', 'Action density', 'Action density', 'Action density', 'Action density'
				type_of_soft_start = 1
				soft_time_interval = 0.0
				reference_values = 0.0, 8.0, 270.0, 5.0, 0.0, 16.0, 270.0, 32.0
				type_of_time_interpolation = 1, 1, 1, 1, 1, 1, 1, 1
				type_of_space_interpolation = 1
				code_cyclic = 0
				reflection_coefficient = 1.0
				type_of_frequency_spectrum = 2
				type_of_frequency_normalization = 1
				sigma_a = 0.07
				sigma_b = 0.09
				gamma = 3.3
				type_of_directional_distribution = 1
				type_of_directional_normalization = 1
				type_of_frequency_spectrum_swell = 2
				type_of_frequency_normalization_swell = 1
				sigma_a_swell = 0.07
				sigma_b_swell = 0.09
				gamma_swell = 5.0
				type_of_directional_distribution_swell = 1
				type_of_directional_normalization_swell = 1
			EndSect  // CODE_3

			[CODE_5]
				touched = 1
				type = 4
				format = 3
				constant_values = 1.0, 8.0, 270.0, 5.0, 0.1, 16.0, 270.0, 32.0
				file_name = |..\input\SountBC_Line.dfs1|
				item_numbers = 1, 2, 3, 4, 1, 1, 1, 1
				item_names = 'Sign. Wave Height', 'Peak Wave Period', 'Mean Wave Direction', 'Dir. Stand. Deviation', 'Sign. Wave Height', 'Sign. Wave Height', 'Sign. Wave Height', 'Sign. Wave Height'
				type_of_soft_start = 1
				soft_time_interval = 0.0
				reference_values = 0.0, 8.0, 270.0, 5.0, 0.0, 16.0, 270.0, 32.0
				type_of_time_interpolation = 1, 1, 1, 1, 1, 1, 1, 1
				type_of_space_interpolation = 1
				code_cyclic = 0
				reflection_coefficient = 1.0
				type_of_frequency_spectrum = 2
				type_of_frequency_normalization = 1
				sigma_a = 0.07
				sigma_b = 0.09
				gamma = 3.3
				type_of_directional_distribution = 1
				type_of_directional_normalization = 1
				type_of_frequency_spectrum_swell = 2
				type_of_frequency_normalization_swell = 1
				sigma_a_swell = 0.07
				sigma_b_swell = 0.09
				gamma_swell = 5.0
				type_of_directional_distribution_swell = 1
				type_of_directional_normalization_swell = 1
			EndSect  // CODE_5

		EndSect  // BOUNDARY_CONDITIONS

		[OUTPUTS]
			touched = 1
			mzsepfslistitemcount = 10
			number_of_outputs = 10
			[OUTPUT_1]
				touched = 1
				include = 0
				title = 'Area'
				file_name = 'SW_Area_2017.dfsu'
				type = 1
				format = 2
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 1
				use_end_time = 1
				first_time_step = 36
				last_time_step = 4320
				time_step_frequency = 18
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 2.95
					y = 51.9363335
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -0.2
						y = 49.872667
					EndSect  // POINT_1

					[POINT_2]
						x = 6.1
						y = 54.0
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -97.4107883362347
					z_ds = 15.56245389898787
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.033
					maximum_frequency = 1.061989502698701
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
						significant_wave_height = 1
						maximum_wave_height = 1
						peak_wave_period = 1
						wave_period_t01 = 1
						wave_period_t02 = 1
						wave_period_tm10 = 0
						peak_wave_direction = 1
						mean_wave_direction = 1
						directional_standard_deviation = 1
						wave_velocity_components = 1
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 1
					water_depth = 0
					current_velocity_components = 1
					wind_speed = 1
					wind_direction = 1
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
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
				touched = 1
				include = 1
				title = 'points'
				file_name = 'ts.dfs0'
				type = 1
				format = 0
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 1
				use_end_time = 1
				first_time_step = 144
				last_time_step = 4320
				time_step_frequency = 1
				number_of_points = 10
				[POINT_1]
					name = 'Europlatform'
					x = 3.276
					y = 51.999
				EndSect  // POINT_1

				[POINT_2]
					name = 'K14'
					x = 3.6333
					y = 53.2667
				EndSect  // POINT_2

				[POINT_3]
					name = 'F16'
					x = 4.0122
					y = 54.1167
				EndSect  // POINT_3

				[POINT_4]
					name = 'HKNA'
					x = 4.242
					y = 52.6887
				EndSect  // POINT_4

				[POINT_5]
					name = 'K13a'
					x = 3.2217
					y = 53.2192
				EndSect  // POINT_5

				[POINT_6]
					name = 'LEG'
					x = 3.6694
					y = 51.929
				EndSect  // POINT_6

				[POINT_7]
					name = 'J6'
					x = 2.9416
					y = 53.8193
				EndSect  // POINT_7

				[POINT_8]
					name = 'L9'
					x = 4.9667
					y = 53.6167
				EndSect  // POINT_8

				[POINT_9]
					name = 'F3'
					x = 4.6939
					y = 54.8489
				EndSect  // POINT_9

				[POINT_10]
					name = 'HKZA'
					x = 4.009
					y = 52.3066
				EndSect  // POINT_10

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -0.2
						y = 49.87266744109285
					EndSect  // POINT_1

					[POINT_2]
						x = 6.1
						y = 54.00000000000001
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -0.263
						y = 49.83139411550378
					EndSect  // POINT_1

					[POINT_2]
						x = -0.263
						y = 54.04127332558908
					EndSect  // POINT_2

					[POINT_3]
						x = 6.162999999999999
						y = 54.04127332558908
					EndSect  // POINT_3

					[POINT_4]
						x = 6.162999999999999
						y = 49.83139411550378
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -97.4107883362347
					z_ds = 15.56245389898787
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.033
					maximum_frequency = 1.061989502698701
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
						significant_wave_height = 1
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 1
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
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
				touched = 1
				include = 0
				title = 'points_eval'
				file_name = 'ts_evaluation.dfs0'
				type = 1
				format = 0
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 1
				use_end_time = 1
				first_time_step = 144
				last_time_step = 4320
				time_step_frequency = 1
				number_of_points = 4
				[POINT_1]
					name = 'K14'
					x = 3.6333
					y = 53.2667
				EndSect  // POINT_1

				[POINT_2]
					name = 'F16'
					x = 4.0122
					y = 54.1167
				EndSect  // POINT_2

				[POINT_3]
					name = 'L9'
					x = 4.9667
					y = 53.6167
				EndSect  // POINT_3

				[POINT_4]
					name = 'HKZA'
					x = 4.009
					y = 52.3066
				EndSect  // POINT_4

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -0.2
						y = 49.87266744109285
					EndSect  // POINT_1

					[POINT_2]
						x = 6.1
						y = 54.00000000000001
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -0.263
						y = 49.83139411550378
					EndSect  // POINT_1

					[POINT_2]
						x = -0.263
						y = 54.04127332558908
					EndSect  // POINT_2

					[POINT_3]
						x = 6.162999999999999
						y = 54.04127332558908
					EndSect  // POINT_3

					[POINT_4]
						x = 6.162999999999999
						y = 49.83139411550378
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -97.4107883362347
					z_ds = 15.56245389898787
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.033
					maximum_frequency = 1.061989502698701
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
						significant_wave_height = 1
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 1
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
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
				touched = 1
				include = 0
				title = 'HKZN'
				file_name = 'HKZN_OWF_area_2002.dfsu'
				type = 1
				format = 2
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 1
				use_end_time = 0
				first_time_step = 0
				last_time_step = 49
				time_step_frequency = 1
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 2.933333333333335
					y = 51.93633372054645
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -0.2
						y = 49.8726674410929
					EndSect  // POINT_1

					[POINT_2]
						x = 6.06666666666667
						y = 54.0
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = 3.0
						y = 50.5
					EndSect  // POINT_1

					[POINT_2]
						x = 2.5
						y = 52.2
					EndSect  // POINT_2

					[POINT_3]
						x = 2.5
						y = 53.5
					EndSect  // POINT_3

					[POINT_4]
						x = 5.5
						y = 53.5
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -97.4107883362347
					z_ds = 15.56245389898787
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.033
					maximum_frequency = 1.061989502698701
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
						significant_wave_height = 1
						maximum_wave_height = 1
						peak_wave_period = 1
						wave_period_t01 = 1
						wave_period_t02 = 1
						wave_period_tm10 = 0
						peak_wave_direction = 1
						mean_wave_direction = 1
						directional_standard_deviation = 1
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 1
						maximum_wave_height = 1
						peak_wave_period = 1
						wave_period_t01 = 1
						wave_period_t02 = 1
						wave_period_tm10 = 0
						peak_wave_direction = 1
						mean_wave_direction = 1
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 1
						maximum_wave_height = 1
						peak_wave_period = 1
						wave_period_t01 = 1
						wave_period_t02 = 1
						wave_period_tm10 = 0
						peak_wave_direction = 1
						mean_wave_direction = 1
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 1
					water_depth = 0
					current_velocity_components = 1
					wind_speed = 1
					wind_direction = 1
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
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

			[OUTPUT_5]
				touched = 1
				include = 0
				title = 'AreaHD'
				file_name = 'areaHD.dfsu'
				type = 1
				format = 2
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 1
				use_end_time = 1
				first_time_step = 0
				last_time_step = 4320
				time_step_frequency = 3
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 3.636925703276114
					y = 52.59992999914525
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -1.578551064219712
						y = 49.87266744109285
					EndSect  // POINT_1

					[POINT_2]
						x = 8.85240247077194
						y = 55.32719255719765
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -93.39006166302535
					z_ds = 14.69730550458667
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.055
					maximum_frequency = 0.5959088268863617
					separation_of_wind_sea_and_swell = 5
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					hm0_minimum = 0.01
					type_of_h_max = 3
					duration = 10800.0
					distance_above_bed_for_particle_velocity = 0.0
					minimum_direction = 0.0
					maximum_direction = 360.0
					[Total_wave_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 1
					water_depth = 0
					current_velocity_components = 1
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
					separation_of_wind_sea_and_swell = 5.0
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					wave_energy = 1
					wave_action = 0
					zeroth_moment_of_wave_action = 0
					first_moment_of_wave_action = 0
					wave_energy_wind_sea = 0
					wave_energy_swell = 0
				EndSect  // SPECTRAL_PARAMETERS

			EndSect  // OUTPUT_5

			[OUTPUT_6]
				touched = 1
				include = 0
				title = 'AreaWind'
				file_name = 'areaWind.dfsu'
				type = 1
				format = 2
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 1
				use_end_time = 1
				first_time_step = 0
				last_time_step = 4320
				time_step_frequency = 3
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 3.636925703276114
					y = 52.59992999914525
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -1.578551064219712
						y = 49.87266744109285
					EndSect  // POINT_1

					[POINT_2]
						x = 8.85240247077194
						y = 55.32719255719765
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -93.39006166302535
					z_ds = 14.69730550458667
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.055
					maximum_frequency = 0.5959088268863617
					separation_of_wind_sea_and_swell = 5
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					hm0_minimum = 0.01
					type_of_h_max = 3
					duration = 10800.0
					distance_above_bed_for_particle_velocity = 0.0
					minimum_direction = 0.0
					maximum_direction = 360.0
					[Total_wave_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 1
					wind_direction = 1
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
					separation_of_wind_sea_and_swell = 5.0
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					wave_energy = 1
					wave_action = 0
					zeroth_moment_of_wave_action = 0
					first_moment_of_wave_action = 0
					wave_energy_wind_sea = 0
					wave_energy_swell = 0
				EndSect  // SPECTRAL_PARAMETERS

			EndSect  // OUTPUT_6

			[OUTPUT_7]
				touched = 1
				include = 0
				title = 'NorthBCpt'
				file_name = 'north_wave_params.dfs0'
				type = 1
				format = 0
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 2
				use_end_time = 1
				first_time_step = 0
				last_time_step = 4320
				time_step_frequency = 1
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 3.636925703276114
					y = 55.2
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -1.578551064219712
						y = 49.87266744109285
					EndSect  // POINT_1

					[POINT_2]
						x = 8.85240247077194
						y = 55.32719255719765
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -93.39006166302535
					z_ds = 14.69730550458667
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.055
					maximum_frequency = 0.5959088268863617
					separation_of_wind_sea_and_swell = 5
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					hm0_minimum = 0.01
					type_of_h_max = 3
					duration = 10800.0
					distance_above_bed_for_particle_velocity = 0.0
					minimum_direction = 0.0
					maximum_direction = 360.0
					[Total_wave_parameters]
						significant_wave_height = 1
						maximum_wave_height = 0
						peak_wave_period = 1
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 1
						directional_standard_deviation = 1
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
					separation_of_wind_sea_and_swell = 5.0
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					wave_energy = 1
					wave_action = 0
					zeroth_moment_of_wave_action = 0
					first_moment_of_wave_action = 0
					wave_energy_wind_sea = 0
					wave_energy_swell = 0
				EndSect  // SPECTRAL_PARAMETERS

			EndSect  // OUTPUT_7

			[OUTPUT_8]
				touched = 1
				include = 0
				title = 'SouthBCpt'
				file_name = 'south_wave_params.dfs0'
				type = 1
				format = 0
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 2
				use_end_time = 1
				first_time_step = 0
				last_time_step = 4320
				time_step_frequency = 1
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 0.7
					y = 50.3
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = -1.578551064219712
						y = 49.87266744109285
					EndSect  // POINT_1

					[POINT_2]
						x = 8.85240247077194
						y = 55.32719255719765
					EndSect  // POINT_2

					npoints = 3
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -93.39006166302535
					z_ds = 14.69730550458667
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.055
					maximum_frequency = 0.5959088268863617
					separation_of_wind_sea_and_swell = 5
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					hm0_minimum = 0.01
					type_of_h_max = 3
					duration = 10800.0
					distance_above_bed_for_particle_velocity = 0.0
					minimum_direction = 0.0
					maximum_direction = 360.0
					[Total_wave_parameters]
						significant_wave_height = 1
						maximum_wave_height = 0
						peak_wave_period = 1
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 1
						directional_standard_deviation = 1
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
					separation_of_wind_sea_and_swell = 5.0
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					wave_energy = 1
					wave_action = 0
					zeroth_moment_of_wave_action = 0
					first_moment_of_wave_action = 0
					wave_energy_wind_sea = 0
					wave_energy_swell = 0
				EndSect  // SPECTRAL_PARAMETERS

			EndSect  // OUTPUT_8

			[OUTPUT_9]
				touched = 1
				include = 0
				title = 'NorthBC'
				file_name = 'NorthBC_Line.dfs1'
				type = 1
				format = 1
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 2
				use_end_time = 1
				first_time_step = 0
				last_time_step = 4320
				time_step_frequency = 3
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 3.636925703276114
					y = 52.59992999914525
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = 8.6755
						y = 55.326
					EndSect  // POINT_1

					[POINT_2]
						x = -1.575
						y = 55.3215
					EndSect  // POINT_2

					npoints = 30
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -93.39006166302535
					z_ds = 14.69730550458667
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.055
					maximum_frequency = 0.5959088268863617
					separation_of_wind_sea_and_swell = 5
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					hm0_minimum = 0.01
					type_of_h_max = 3
					duration = 10800.0
					distance_above_bed_for_particle_velocity = 0.0
					minimum_direction = 0.0
					maximum_direction = 360.0
					[Total_wave_parameters]
						significant_wave_height = 1
						maximum_wave_height = 0
						peak_wave_period = 1
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 1
						directional_standard_deviation = 1
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
					separation_of_wind_sea_and_swell = 5.0
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					wave_energy = 1
					wave_action = 0
					zeroth_moment_of_wave_action = 0
					first_moment_of_wave_action = 0
					wave_energy_wind_sea = 0
					wave_energy_swell = 0
				EndSect  // SPECTRAL_PARAMETERS

			EndSect  // OUTPUT_9

			[OUTPUT_10]
				touched = 1
				include = 0
				title = 'SouthBC'
				file_name = 'SountBC_Line.dfs1'
				type = 1
				format = 1
				flood_and_dry = 2
				coordinate_type = 'LONG/LAT'
				zone = 0
				input_file_name = None
				input_format = 1
				interpolation_type = 2
				use_end_time = 1
				first_time_step = 0
				last_time_step = 4320
				time_step_frequency = 3
				number_of_points = 1
				[POINT_1]
					name = 'Point 1'
					x = 3.636925703276114
					y = 52.59992999914525
				EndSect  // POINT_1

				[LINE]
					number_of_points = 2
					[POINT_1]
						x = 0.29
						y = 50.7565
					EndSect  // POINT_1

					[POINT_2]
						x = 0.7165
						y = 49.87268
					EndSect  // POINT_2

					npoints = 10
				EndSect  // LINE

				[AREA]
					number_of_points = 4
					[POINT_1]
						x = -1.682860599569629
						y = 49.81812218993181
					EndSect  // POINT_1

					[POINT_2]
						x = -1.682860599569629
						y = 55.38173780835869
					EndSect  // POINT_2

					[POINT_3]
						x = 8.956712006121856
						y = 55.38173780835869
					EndSect  // POINT_3

					[POINT_4]
						x = 8.956712006121856
						y = 49.81812218993181
					EndSect  // POINT_4

					orientation = 0.0
					x_origo = -1.578551064219712
					x_ds = 0.5489975544732448
					x_npoints = 20
					y_origo = 49.87266744109285
					y_ds = 0.5489975544732448
					y_npoints = 11
					z_origo = -93.39006166302535
					z_ds = 14.69730550458667
					z_npoints = 10
				EndSect  // AREA

				[INTEGRAL_WAVE_PARAMETERS]
					touched = 0
					type_of_spectrum = 1
					minimum_frequency = 0.055
					maximum_frequency = 0.5959088268863617
					separation_of_wind_sea_and_swell = 5
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					hm0_minimum = 0.01
					type_of_h_max = 3
					duration = 10800.0
					distance_above_bed_for_particle_velocity = 0.0
					minimum_direction = 0.0
					maximum_direction = 360.0
					[Total_wave_parameters]
						significant_wave_height = 1
						maximum_wave_height = 0
						peak_wave_period = 1
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 1
						directional_standard_deviation = 1
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Total_wave_parameters

					[Wind_sea_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Wind_sea_parameters

					[Swell_parameters]
						significant_wave_height = 0
						maximum_wave_height = 0
						peak_wave_period = 0
						wave_period_t01 = 0
						wave_period_t02 = 0
						wave_period_tm10 = 0
						peak_wave_direction = 0
						mean_wave_direction = 0
						directional_standard_deviation = 0
						wave_velocity_components = 0
						radiation_stresses = 0
						particle_velocities = 0
						wave_power = 0
					EndSect  // Swell_parameters

				EndSect  // INTEGRAL_WAVE_PARAMETERS

				[INPUT_PARAMETERS]
					touched = 0
					surface_elevation = 0
					water_depth = 0
					current_velocity_components = 0
					wind_speed = 0
					wind_direction = 0
					ice_concentration = 0
				EndSect  // INPUT_PARAMETERS

				[MODEL_PARAMETERS]
					touched = 0
					wind_friction_speed = 0
					roughness_length = 0
					drag_coefficient = 0
					charnock_constant = 0
					friction_coefficient = 0
					breaking_parameter_gamma = 0
					courant_number = 0
					time_step_factor = 0
					convergence_angle = 0
					length = 0
					area = 0
					threshold_period = 0
					roller_area = 0
					roller_dissipation = 0
					breaking_index = 0
				EndSect  // MODEL_PARAMETERS

				[SPECTRAL_PARAMETERS]
					touched = 0
					separation_of_wind_sea_and_swell = 5.0
					threshold_frequency = 0.125
					maximum_threshold_frequency = 0.125
					wave_energy = 1
					wave_action = 0
					zeroth_moment_of_wave_action = 0
					first_moment_of_wave_action = 0
					wave_energy_wind_sea = 0
					wave_energy_swell = 0
				EndSect  // SPECTRAL_PARAMETERS

			EndSect  // OUTPUT_10

		EndSect  // OUTPUTS

	EndSect  // SPECTRAL_WAVE_MODULE

EndSect