from typing import Callable, Dict, Tuple, Optional, Any, NamedTuple
import os
import json
import warnings
from functools import partial
import glob

from flax.core import FrozenDict
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from jaxfluids_rl.helper_functions import fig_to_img
import numpy as np

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.ml_buffers import (
    ParametersSetup, CallablesSetup, LevelSetSetup,
    BoundaryConditionSetup, BoundaryConditionsField, BoundaryConditionsFace
)

from jaxfluids_rl.helper_functions import get_advance_fn

Array = jax.Array


class Channel3DEnv(gym.Env):

    # TODO set reasonable bounds (and Reynolds dependent bounds)
    LOW_ACTION = -0.1
    HIGH_ACTION = 0.1
    LOW_OBSERVATION = 0.0
    HIGH_OBSERVATION = 1.0

    REWARD_NAN = -1e2

    RESOLUTION_DICT = {
        180: {
            "minimal": (32, 96, 32),
            "full": (96, 96, 96)
        },
        395: {
            "minimal": (72, 128, 72),
            "full": (256, 128, 256)
        },
        590: {
            "minimal": (128, 192, 128),
            "full": (384, 192, 384)
        },
    }

    VISCOSITY_DICT = {
        180: 2.0 / 5600.0,
        395: 2.0 / 13750.0,
        590: 2.0 / 21950.0,
    }

    DOMAIN_DICT = {
        "minimal": [(0.0, 1.77), (-1.0, 1.0), (0.0, 0.89)],
        "full"   : [(0.0, 2*np.pi), (-1.0, 1.0), (0.0, np.pi)],
    }

    RENDER_MODES = ("save", "show", "export")

    def __init__(
            self,
            args,
            episode_length: Optional[int] = 1.0,
            action_length: Optional[float] = 0.1,
            reynolds_number: int = 180,
            channel_type: str = "minimal",
            observation_type: str = "asymmetric",
            render_mode: Optional[str] = "save",
            is_double_precision: Optional[bool] = True,
        ) -> None:

        self.args =  args
        self.end_time = episode_length
        self.action_time = action_length

        assert_str = "channel_type must either be 'minimal' or 'full'."
        assert channel_type in ("minimal", "full"), assert_str
        self.channel_type = channel_type

        assert_str = "reynolds_number must be in (180, 395, 590)."
        assert reynolds_number in (180, 395, 590), assert_str
        self.reynolds_number = reynolds_number

        assert_str = "observation_type must be 'partial' or 'asymmetric'."
        assert observation_type in ("partial", "asymmetric"), assert_str
        self.observation_type = observation_type

        assert_str = f"render_mode must be one of {self.RENDER_MODES} or None"
        assert render_mode in self.RENDER_MODES or render_mode is None, assert_str
        self.render_mode = render_mode

        domain_size = self.DOMAIN_DICT[channel_type]
        resolution = self.RESOLUTION_DICT[reynolds_number][channel_type]
        viscosity_ref = self.VISCOSITY_DICT[reynolds_number]

        dirname = os.path.dirname(os.path.realpath(__file__))
        inputfiles_path = os.path.join(dirname, "inputfiles")
        case_setup_path = os.path.join(inputfiles_path, "case_setup.json")
        numerical_setup_path = os.path.join(inputfiles_path, "numerical_setup.json")

        # NOTE Modify case setup
        case_setup = json.load(open(case_setup_path, "r"))

        case_setup["general"]["end_time"] = self.end_time

        # NOTE Set restart file
        # TODO select random restart file and adjust for Reynolds number
        restart_files = glob.glob(os.path.join(dirname, f"restart_files/Re{reynolds_number}/data_*.h5"))
        times = []
        for file in restart_files:
            times_i = file.split("_")[-1]
            times_i = os.path.splitext(times_i)[0]
            times.append(float(times_i))
        indices = np.argsort(np.array(times))
        self.restart_files = list(np.array(restart_files)[indices])

        case_setup["restart"] = {
            "flag": True,
            "file_path": self.restart_files[0],
            "use_time": True,
            "time": 0.0
        }

        # NOTE Set extent and resolution of the computational domain
        for i, xi in enumerate(("x", "y", "z")):
            case_setup["domain"][xi]["cells"] = resolution[i]
            case_setup["domain"][xi]["range"] = list(domain_size[i])

        # NOTE Set boundary conditions at walls
        for wall in ("north", "south"):
            case_setup["boundary_conditions"][wall]["type"] = "MASSTRANSFERWALL_PARAMETERIZED"
            case_setup["boundary_conditions"][wall]["wall_mass_transfer"] = {
                "primitives_callable": {
                    "rho": 1.0,
                    "u": 0.0,
                    "v": 0.0,
                    "w": 0.0
                },
                "bounding_domain": "lambda x,z: jnp.ones_like(x)"
            }


        # NOTE Set Reynolds number by adjusting the dynamic viscosity
        case_setup["material_properties"]["transport"]["dynamic_viscosity"]["value"] = f"lambda T: {viscosity_ref} * T**0.75" 

        # NOTE Modify numerical setup
        numerical_setup = json.load(open(numerical_setup_path, "r"))
        numerical_setup["precision"]["is_double_precision_compute"] = is_double_precision 
        numerical_setup["precision"]["is_double_precision_output"] = is_double_precision 

        self.input_manager = InputManager(case_setup, numerical_setup)
        self.init_manager = InitializationManager(self.input_manager)
        self.sim_manager = SimulationManager(
            self.input_manager,
            callbacks=None
        )

        self.domain_information = self.sim_manager.domain_information
        self.equation_information = self.sim_manager.equation_information

        self.advance_fn = get_advance_fn(self.sim_manager)

        # NOTE Set action_space and observation_space
        is_double_precision = self.sim_manager.numerical_setup.precision.is_double_precision_compute
        dtype = np.float64 if is_double_precision else np.float32
        Nx, Ny, Nz = resolution

        self.action_space = gym.spaces.Box(
            low=self.LOW_ACTION, high=self.HIGH_ACTION, shape=(2,Nx,Nz), dtype=dtype
        )
        
        if self.observation_type == "full":
            self.observation_space = gym.spaces.Box(
                    low=self.LOW_OBSERVATION, high=self.HIGH_OBSERVATION, shape=(5,Nx,Ny,Nz), dtype=dtype
                )
        elif self.observation_type == "partial":
            self.observation_space = gym.spaces.Box(
                low=self.LOW_OBSERVATION, high=self.HIGH_OBSERVATION, shape=(2,Nx,Nz), dtype=dtype
            )
        else:
            raise NotImplementedError

        class PrimitivesCallable(NamedTuple):
            rho: Callable
            u: Callable
            v: Callable
            w: Callable

        primitives_callable = PrimitivesCallable(
        	rho=lambda x,z,t,params: jnp.ones_like(x),
        	u=lambda x,z,t,params: jnp.zeros_like(x), 
        	v=lambda x,z,t,params: params, 
        	w=lambda x,z,t,params: jnp.zeros_like(x),
        )

        self.ml_callables = CallablesSetup(
            boundary_conditions=BoundaryConditionSetup(
                primitives=BoundaryConditionsField(
                    south=(
                        BoundaryConditionsFace(
                            wall_mass_transfer=primitives_callable
                        ),
                    ),
                    north=(
                        BoundaryConditionsFace(
                            wall_mass_transfer=primitives_callable
                        ),
                    ),
                )
            )
        )

    def step(self, action_south: np.ndarray | Array,action_north: np.ndarray | Array) -> Tuple[Array, float, bool, bool, Dict]:

        action_north = np.asarray(action_north).copy()
        action_south = np.asarray(action_south).copy()

        jxf_buffers, cb_buffers = self.state
        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time
        end_time = physical_simulation_time + self.action_time

        ml_parameters = self._wrap_action_for_jxf(action_south, action_north)
        ml_callables = self.ml_callables
        
        # NOTE Time advance jxf_buffers
        (
            jxf_buffers,
            cb_buffers,
            wall_clock_times
        ) = self.advance_fn(
            jxf_buffers=jxf_buffers,
            ml_parameters=ml_parameters,
            ml_callables=ml_callables,
            end_time=end_time,
            end_step=int(1e+8)
        )

        jxf_buffers: JaxFluidsBuffers

        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time      

        self.state = (jxf_buffers, cb_buffers)

        observation = self._get_obs()

        pressure_bottom, pressure_top = observation[0],observation[1]

        reward = self._get_reward(action_top,action_bottom)
        
        terminated = physical_simulation_time > self.end_time

        truncated = False

        info = self._get_info()

        if not np.isfinite(reward):
            terminated = True
            reward = self.REWARD_NAN
            info["success"] = 0
        else:
            info["success"] = 1

        return pressure_bottom, reward, terminated ,truncated, info
    
    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None
        ) -> Tuple[Array, Dict]:
        super().reset(seed=seed)

        restart_file = np.random.choice(self.restart_files)

        # TODO implement random burn-in time

        action = np.zeros(self.action_space.shape)
        ml_parameters = self._wrap_action_for_jxf(action[0],action[1])
        ml_callables = self.ml_callables

        jxf_buffers = self.init_manager.initialization(
            ml_parameters=ml_parameters,
            ml_callables=ml_callables
        )
        self.state = (jxf_buffers, None)

        observation = self._get_obs()
        info = self._get_info()

        return observation

    def render(self) -> None:

        if self.render_mode in self.RENDER_MODES:
            
            domain_information = self.sim_manager.domain_information
            equation_information = self.sim_manager.equation_information

            nhx, nhy, nhz = domain_information.domain_slices_conservatives

            ids_velocity = equation_information.ids_velocity
            ids_energy = equation_information.ids_energy

            jxf_buffers, _ = self.state

            primitives = jxf_buffers.simulation_buffers.material_fields.primitives
            primitives = primitives[..., nhx, nhy, nhz]


            X, Y, Z = domain_information.compute_mesh_grid()

            fig = plt.figure(layout="constrained", figsize=(30,10))

            gs = GridSpec(7, 3, figure=fig)

            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            ax2 = fig.add_subplot(gs[2,:])

            ax3 = fig.add_subplot(gs[3,:])
            ax4 = fig.add_subplot(gs[4,:])

            ax5 = fig.add_subplot(gs[5,:])
            ax6 = fig.add_subplot(gs[6,:])

            ax0.pcolormesh(X[:,0,:], Z[:,0,:], primitives[ids_velocity[0],:,0,:])
            ax1.pcolormesh(X[:,0,:], Z[:,0,:], primitives[ids_velocity[1],:,0,:])
            ax2.pcolormesh(X[:,0,:], Z[:,0,:], primitives[ids_velocity[2],:,0,:])

            ax3.pcolormesh(X[:,0,:], Z[:,0,:], primitives[ids_energy,:,0,:])
            ax4.pcolormesh(X[:,0,:], Z[:,0,:], primitives[ids_energy,:,-1,:])

            ax5.pcolormesh(X[:,0,:], Z[:,0,:], self.last_action[0,:,:], vmin=self.LOW_ACTION, vmax=self.HIGH_ACTION)
            ax6.pcolormesh(X[:,0,:], Z[:,0,:], self.last_action[1,:,:], vmin=self.LOW_ACTION, vmax=self.HIGH_ACTION)

            ax = (ax0, ax1, ax2, ax3, ax4, ax5, ax6)
            titles = (r"$u$", r"$v$", r"$w$", r"$p_{W,bottom}$", r"$p_{W,top}$", r"$a_{bottom}$", r"$a_{top}$")
            for ax_i, title_i in zip(ax, titles):
                ax_i.set_xlabel(r"$x$")
                ax_i.set_title(title_i)
                ax_i.set_aspect("equal")

            for ax_i in (ax0, ax1, ax2):
                ax_i.set_ylabel(r"$z$")

            for ax_i in (ax3, ax4, ax5, ax6):
                ax_i.set_ylabel(r"$z$")

            if self.render_mode == "save":
                plt.savefig("Channel3DEnv_render0.png", bbox_inches="tight", dpi=400)
                plt.close()
            elif self.render_mode == "export":
                img = fig_to_img(fig)
                plt.close()
                return img
            else:
                plt.show()

        else:
            pass

    def close(self) -> None:
        pass


    def _get_partial_obs(self, primitives: Array) -> Array:

        # TODO parallel

        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives

        equation_information = self.equation_information
        id_pressure = equation_information.ids_energy

        pressure = primitives[id_pressure,nhx,nhy,nhz]
        wall_pressure = jnp.moveaxis(pressure[:,(0,-1),:], 1, 0)
        

        return wall_pressure


    def _get_full_obs(self, primitives: Array) -> Array:

        # TODO parallel
        domain_information = self.domain_information
        nhx, nhy, nhz = domain_information.domain_slices_conservatives

        return primitives[..., nhx, nhy, nhz]


    def _get_obs(self) -> Array:
        
        obs_fn = None
        if self.observation_type == "full":
            obs_fn = lambda p: self._get_full_obs(p)
        elif self.observation_type == "partial":
            obs_fn = lambda p: self._get_partial_obs(p)
        else:
            raise NotImplementedError

        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives

        obs = jax.jit(obs_fn)(primitives)

        return obs
    
    def get_boundary_pressures(self):
        pressures = self._get_obs()
        bottom_pressure = pressures[0]
        top_pressure = pressures[1]
        return bottom_pressure, top_pressure
    
    def calculate_divergence(self, primitives: Array) -> Array:
        """
        Calculate the divergence of the velocity field.
        """
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        ids_velocity = self.equation_information.ids_velocity
        dx, dy, dz = self.domain_information.get_device_cell_sizes()

        u = primitives[ids_velocity[0], nhx, nhy, nhz]
        v = primitives[ids_velocity[1], nhx, nhy, nhz]
        w = primitives[ids_velocity[2], nhx, nhy, nhz]

        du_dx = (u[:, (1, -2), :] - u[:, (0, -1), :]) / dx[:,(0,-1),:]
        dv_dy = (v[:, (1, -2), :] - v[:, (0, -1), :]) / dy[:,(0,-1),:]
        dw_dz = (w[:, (1, -2), :] - w[:, (0, -1), :]) / dz[:,(0,-1),:]

        divergence = du_dx + dv_dy + dw_dz
        return jnp.moveaxis(divergence, 1, 0)

    def _get_reward(self, action: Array) -> Array:

        def compute_reward(primitives: Array, action: Array) -> float:
            
            nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
            s_velocity = self.equation_information.s_velocity            
            velocity = primitives[s_velocity, nhx, nhy, nhz]

            # NOTE Compute mean velocity 
            velocity_mean = jnp.mean(velocity, axis=(-1,-3), keepdims=True)
            is_parallel = self.domain_information.is_parallel
            if is_parallel:
                velocity_mean = jax.lax.pmean(velocity_mean, axis_name="i")

            # NOTE Compute fluctuating velocity (velocity_prime) and
            # turbulent kinetic energy (tke)
            velocity_prime = velocity - velocity_mean
            tke = jnp.sum(jnp.square(velocity_prime), axis=0)
            dx, dy, dz = self.domain_information.get_device_cell_sizes()
            tke = jnp.sum(tke * dx * dz, axis=(-1,-3), keepdims=True)
            tke = jnp.sum(tke * dy, axis=-2, keepdims=True)
            if is_parallel:
                tke = jax.lax.psum(tke, axis_name="i")

            lambda_n = 0.
            cost = tke + lambda_n * jnp.sum(jnp.square(action))
            reward = -jnp.squeeze(cost)

            return 100*reward


        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives

        reward = jax.jit(compute_reward)(primitives, action)

        return reward
    

    def _calculate_du_dy(self,primitives: Array):
        """
        Calculate the du/dy at the wall.
        """
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        ids_velocity = self.equation_information.ids_velocity
        dx, dy, dz = self.domain_information.get_device_cell_sizes()
        u = primitives[ids_velocity[0], nhx, nhy, nhz]
        dudy_wall = (u[:, (1, -2), :] - u[:, (0, -1), :]) / dy[:,(0,-1),:]
        dudy_wall = jnp.moveaxis(dudy_wall, 1, 0)
        return dudy_wall

    def pressure_gradient(self):
        """
        Calculate the pressure gradient at the wall.
        """
        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        ids_energy = self.equation_information.ids_energy
        pressure = primitives[ids_energy, nhx, nhy, nhz]
        pressure_gradient = (pressure[:, (1, -2), :] - pressure[:, (0, -1), :]) / self.domain_information.get_device_cell_sizes()[0]
        pressure_gradient = jnp.moveaxis(pressure_gradient, 1, 0)
        return  jnp.abs(jnp.mean(pressure_gradient))
    
    def _get_shear_stress(self):
        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        ids_velocity = self.equation_information.ids_velocity
        u = primitives[ids_velocity[0], nhx, nhy, nhz]
        v = primitives[ids_velocity[1], nhx, nhy, nhz]

        wall_u = jnp.moveaxis(u[:, (0, -1), :], 1, 0)
        wall_v = jnp.moveaxis(v[:, (0, -1), :], 1, 0)

        dudy_wall = self._calculate_du_dy(primitives)
        shear_stress = -wall_u * wall_v + self.VISCOSITY_DICT[self.reynolds_number] * dudy_wall
        shear_stress_mean = jnp.mean(shear_stress, axis=(-1,-2), keepdims=True)
        total_shear_stress = jnp.sum(shear_stress_mean)

        return jnp.abs(total_shear_stress).item()


    def _get_info(self):

        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives
        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        ids_velocity = self.equation_information.ids_velocity

        shear_stress_residual = self._get_shear_stress()
        u = primitives[ids_velocity[0], nhx, nhy, nhz]




        return {"shear_stress": shear_stress_residual,
                "mean_velocity": jnp.mean(u).item()}
    
    def gt_control(self) -> Array:

        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives
        
        ids_velocity = self.equation_information.ids_velocity
        velocities = self._get_full_obs(primitives)[ids_velocity, ...]
        north_control = -velocities[1, :, self.args.detection_plane, :]
        south_control = -velocities[1, :, -self.args.detection_plane, :]

        return south_control, north_control

    @partial(jax.jit, static_argnums=0)
    def _wrap_action_for_jxf(self, action_south: np.ndarray,action_north: np.ndarray) -> ParametersSetup:

        action_north = jnp.array(action_north)
        action_south = jnp.array(action_south)

        class PrimitivesParameters(NamedTuple):
            rho: Optional[Array]
            u: Optional[Array]
            v: Optional[Array]
            w: Optional[Array]


        parameters_south = PrimitivesParameters(
            rho=None, u=None, v=action_south, w=None
        )
        parameters_north = PrimitivesParameters(
            rho=None, u=None, v=action_north, w=None
        )

        parameter_setup = ParametersSetup(
            boundary_conditions=BoundaryConditionSetup(
                primitives=BoundaryConditionsField(
                    south=(
                        BoundaryConditionsFace(
                            wall_mass_transfer=parameters_south,
                        ),
                    ),
                    north=(
                        BoundaryConditionsFace(
                            wall_mass_transfer=parameters_north,
                        ),
                    )
                )
            )
        )

        return parameter_setup