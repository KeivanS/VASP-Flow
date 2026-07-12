#!/usr/bin/env python3
"""
Instruction Parser Module
Parses plain text instruction files for VASP workflow agent
"""

import re
from typing import Dict, List, Any

class InstructionParser:
    """Parse natural language instructions for VASP calculations"""
    
    def __init__(self, instruction_file: str):
        self.instruction_file = instruction_file
        self.instructions = {}
        self.parse()
    
    def parse(self):
        """Parse the instruction file"""
        with open(self.instruction_file, 'r') as f:
            content = f.read()
        
        self.instructions = {
            'project_name':   self._extract_project_name(content),
            'functional':     self._extract_functional(content),
            'soc':            self._extract_soc(content),
            'magnetization':  self._extract_magnetization(content),
            'gga_u':          self._extract_gga_u(content),
            'tasks':          self._extract_tasks(content),
            'convergence':    self._extract_convergence(content),
            'kpath':          self._extract_kpath(content),
            'wannier':        self._extract_wannier(content),
            'transport':      self._extract_transport(content),
            'dfpt':           self._extract_dfpt(content),
            'phonons':        self._extract_phonons(content),
            'dos_projections': self._extract_dos_projections(content),
            # LOBSTER Gaussian smearing width (eV) for lobsterin; default applied later
            'lobster_sigma':  self._extract_str_key(content,
                              r'LOBSTER_SIGMA\s*[:,=]\s*([\d.]+)', default=None),
            # LOBSTER NSCF symmetry (must be 0 or -1); default applied later
            'lobster_isym':   self._extract_str_key(content,
                              r'LOBSTER_ISYM\s*[:,=]\s*(-?\d+)', default=None),
            # MPI / parallelization (global)
            'mpi_np':         self._extract_int_key(content, r'MPI\s*[:,=]?\s*(\d+)', default=1),
            'kpar':           self._extract_int_key(content, r'(?<!_)KPAR\s*[:,=]\s*(\d+)', default=None),
            'ncore':          self._extract_int_key(content, r'(?<!_)NCORE\s*[:,=]\s*(\d+)', default=None),
            # Per-task KPAR overrides
            'relax_kpar':     self._extract_int_key(content, r'RELAX_KPAR\s*[:,=]\s*(\d+)', default=None),
            'scf_kpar':       self._extract_int_key(content, r'SCF_KPAR\s*[:,=]\s*(\d+)', default=None),
            'bands_kpar':     self._extract_int_key(content, r'BANDS?_KPAR\s*[:,=]\s*(\d+)', default=None),
            'dos_kpar':       self._extract_int_key(content, r'DOS_KPAR\s*[:,=]\s*(\d+)', default=None),
            'dfpt_kpar':      self._extract_int_key(content, r'DFPT_KPAR\s*[:,=]\s*(\d+)', default=None),
            'phonons_kpar':   self._extract_int_key(content, r'PHONONS?_KPAR\s*[:,=]\s*(\d+)', default=None),
            # Per-task NCORE overrides
            'relax_ncore':    self._extract_int_key(content, r'RELAX_NCORE\s*[:,=]\s*(\d+)', default=None),
            'scf_ncore':      self._extract_int_key(content, r'SCF_NCORE\s*[:,=]\s*(\d+)', default=None),
            'bands_ncore':    self._extract_int_key(content, r'BANDS?_NCORE\s*[:,=]\s*(\d+)', default=None),
            'dos_ncore':      self._extract_int_key(content, r'DOS_NCORE\s*[:,=]\s*(\d+)', default=None),
            'dfpt_ncore':     self._extract_int_key(content, r'DFPT_NCORE\s*[:,=]\s*(\d+)', default=None),
            'phonons_ncore':  self._extract_int_key(content, r'PHONONS?_NCORE\s*[:,=]\s*(\d+)', default=None),
            # SLURM / HPC settings (override profile defaults when present)
            'slurm_nodes':          self._extract_int_key(content,
                                    r'NODES?\s*[:,=]\s*(\d+)', default=None),
            'slurm_ntasks_per_node':self._extract_int_key(content,
                                    r'(?:NTASKS?_PER_NODE|TASKS?_PER_NODE|CORES?_PER_NODE)\s*[:,=]\s*(\d+)',
                                    default=None),
            'slurm_partition':      self._extract_str_key(content,
                                    r'PARTITION\s*[:,=]\s*(\S+)', default=None),
            'slurm_walltime':       self._extract_str_key(content,
                                    r'(?:WALLTIME|WALL_TIME|TIME)\s*[:,=]\s*(\S+)', default=None),
            'slurm_account':        self._extract_str_key(content,
                                    r'ACCOUNT\s*[:,=]\s*(\S+)', default=None),
            # ELF (ELFCAR): on by default; disable with "ELF: off" / "no ELF"
            'elf':            self._extract_elf(content),
            # Geometry hints
            'is_2d':          self._extract_bool_key(content, r'\b2[Dd]\b|\bmonolayer\b|\bslab\b'),
            'is_hex':         self._extract_bool_key(content, r'\bhex\w*\b|\btrigonal\b|\bhexagonal\b'),
            'gamma_only':     self._extract_bool_key(content, r'\bgamma[- ]only\b'),
            # Relaxation control
            'isif':           self._extract_int_key(content, r'ISIF\s*[:,=]\s*(\d+)', default=None),
            'nsw':            self._extract_int_key(content, r'(?:NSW|MAX_STEPS)\s*[:,=]\s*(\d+)', default=None),
            'ediffg':         self._extract_float_key(content, r'EDIFFG\s*[:,=]\s*([-\d.Ee]+)', default=None),
            # Direct ENCUT value (not a range — requires = or :)
            'encut_val':      self._extract_int_key(content, r'ENCUT\s*[:,=]\s*(\d+)', default=None),
            # Collinear magnetization: default moment per atom (μB)
            'mag_moment':     self._extract_float_key(content, r'MAGMOM\s*[:,=]\s*([\d.]+)', default=2.0),
            # Band structure resolution
            'nkpts_bands':    self._extract_int_key(content,
                              r'(?:BANDS?_)?NKPTS?\s*[:,=]\s*(\d+)', default=None),
            # Explicit auto-mesh override, e.g. "KMESH: 8 8 8" or "KMESH: 12"
            'kmesh':          self._extract_kmesh(content),
            # Constant-pressure relaxation (ISIF=3, IBRION=2, PSTRESS)
            'pressure':       self._extract_pressure(content),
            # Raw INCAR tags passed straight through (global + per-step)
            'incar_raw':      self._extract_incar_raw(content),
        }

    def _extract_str_key(self, content: str, pattern: str, default=None):
        """Extract a single string token from a regex pattern."""
        m = re.search(pattern, content, re.IGNORECASE)
        return m.group(1).strip() if m else default

    def _extract_int_key(self, content: str, pattern: str, default=None):
        """Extract a single integer from a regex pattern."""
        m = re.search(pattern, content, re.IGNORECASE)
        return int(m.group(1)) if m else default

    def _extract_float_key(self, content: str, pattern: str, default=None):
        """Extract a single float from a regex pattern."""
        m = re.search(pattern, content, re.IGNORECASE)
        return float(m.group(1)) if m else default

    def _extract_bool_key(self, content: str, pattern: str) -> bool:
        """Return True if pattern is found anywhere in content."""
        return bool(re.search(pattern, content, re.IGNORECASE))

    def _extract_elf(self, content: str) -> bool:
        """ELFCAR computation flag. Default ON; disabled by an explicit
        'ELF: off/false/no' key or a 'no ELF' / 'disable ELF' phrase."""
        m = re.search(r'ELF\s*[:=]\s*(on|off|true|false|yes|no)', content, re.IGNORECASE)
        if m:
            return m.group(1).lower() in ('on', 'true', 'yes')
        if re.search(r'\b(no|disable|without)\s+elf\b', content, re.IGNORECASE):
            return False
        return True

    def _extract_project_name(self, content: str) -> str:
        """Extract project name — tolerates stray leading characters before 'Project:'"""
        match = re.search(r'Project:\s*(.+)', content, re.IGNORECASE)
        return match.group(1).strip() if match else "VASP_Calculation"
    
    def _extract_functional(self, content: str) -> str:
        """Extract functional type. More specific keys checked first."""
        # Order matters: most specific first
        functionals = [
            ('pbesol', 'PS'),
            ('r2scan', 'R2SCAN'),
            ('hse06',  'HSE06'),
            ('hse',    'HSE06'),
            ('rvv10',  'VV10'),
            ('am05',   'AM'),
            ('lda',    'LDA'),
            ('pbe',    'PBE'),
        ]
        content_lower = content.lower()
        for key, value in functionals:
            if key in content_lower:
                return value
        return 'PBE'  # default
    
    def _extract_soc(self, content: str) -> bool:
        """Check if SOC is requested"""
        return bool(re.search(r'\bSOC\b', content, re.IGNORECASE))
    
    def _extract_magnetization(self, content: str) -> Dict[str, Any]:
        """Extract magnetization direction"""
        mag_info = {'enabled': False, 'direction': None}
        
        if re.search(r'magnet', content, re.IGNORECASE):
            mag_info['enabled'] = True
            
            # Check for direction
            if re.search(r'z-direction|magnetization\s+in\s+z', content, re.IGNORECASE):
                mag_info['direction'] = 'z'
            elif re.search(r'x-direction|magnetization\s+in\s+x', content, re.IGNORECASE):
                mag_info['direction'] = 'x'
            elif re.search(r'y-direction|magnetization\s+in\s+y', content, re.IGNORECASE):
                mag_info['direction'] = 'y'
        
        return mag_info
    
    def _extract_gga_u(self, content: str) -> Dict[str, Any]:
        """Extract GGA+U parameters"""
        u_info = {'enabled': False, 'elements': {}}
        
        # Look for GGA+U or DFT+U
        if re.search(r'(GGA\+U|DFT\+U)', content, re.IGNORECASE):
            u_info['enabled'] = True
            
            # Extract U values like "U=3.0 on Mo-d"
            u_matches = re.findall(r'U\s*=\s*([0-9.]+)\s+on\s+(\w+)[-\s]*([spdf])?', content, re.IGNORECASE)
            for u_val, element, orbital in u_matches:
                u_info['elements'][element] = {
                    'U': float(u_val),
                    'orbital': orbital if orbital else 'd'
                }
        
        return u_info
    
    def _extract_tasks(self, content: str) -> List[str]:
        """Extract list of tasks to perform"""
        tasks = []
        
        task_keywords = {
            'relax': ['relax', 'optimization', 'structure optimization'],
            'scf': ['scf', 'self-consistent'],
            'bands': ['band structure', 'bands', 'bandstructure'],
            'dos': ['dos', 'density of states'],
            'fatbands': ['fat bands', 'fatbands', 'projected bands'],
            'wannier': ['wannier', 'wannierization', 'wannier90'],
            'transport': ['transport', 'boltzwann', 'boltzmann'],
            'dfpt': ['dfpt', 'born charges', 'born effective', 'dielectric', 'lepsilon'],
            'phonons': ['phonon', 'phonons', 'phonopy', 'vibrational', 'lattice dynamics'],
            'lobster': ['lobster', 'cohp', 'cobi', 'coop', 'crystal orbital',
                        'bonding analysis'],
        }

        content_lower = content.lower()
        for task, keywords in task_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tasks.append(task)

        # LOBSTER analyses need the SCF charge density to start from.
        if 'lobster' in tasks and 'scf' not in tasks:
            tasks.append('scf')

        # Ensure logical order
        task_order = ['relax', 'scf', 'bands', 'dos', 'fatbands', 'wannier', 'transport',
                      'dfpt', 'phonons', 'lobster']
        ordered_tasks = [t for t in task_order if t in tasks]
        
        return ordered_tasks
    
    def _extract_convergence(self, content: str) -> Dict[str, Any]:
        """Extract convergence test parameters.

        Supports two k-point formats:

          Explicit list (preferred for bulk):
            Test k-points 6x6x3, 12x12x6, 18x18x9

          Range (auto-steps, good for 2D):
            Test k-points from 6x6 to 18x18
        """
        conv_info = {
            'kpoints': {'enabled': False, 'meshes': [], 'range': []},
            'encut':   {'enabled': False, 'range': []}
        }

        # ── explicit mesh list: "6x6x3, 12x12x6, 18x18x9" ──────────────
        # Must appear on a k-points convergence line
        kp_line = re.search(r'k-?points?\s+(.+?)(?:\n|$)', content, re.IGNORECASE)
        if kp_line:
            line_text = kp_line.group(1)
            meshes = re.findall(r'(\d+)x(\d+)(?:x(\d+))?', line_text)
            if len(meshes) >= 2 and 'to' not in line_text.lower():
                # Two or more explicit meshes on the same line → list mode
                parsed = []
                for m in meshes:
                    nx, ny = int(m[0]), int(m[1])
                    nz = int(m[2]) if m[2] else ny
                    parsed.append([nx, ny, nz])
                conv_info['kpoints']['enabled'] = True
                conv_info['kpoints']['meshes']  = parsed

        # ── range fallback: "from 6x6x1 to 18x18x1" ────────────────────
        if not conv_info['kpoints']['meshes']:
            kpoint_match = re.search(
                r'k-?points?\s+(?:from\s+)?(\d+)x(\d+)(?:x(\d+))?\s+to\s+(\d+)x(\d+)(?:x(\d+))?',
                content, re.IGNORECASE)
            if kpoint_match:
                conv_info['kpoints']['enabled'] = True
                g = kpoint_match.groups()
                start = [int(g[0]), int(g[1]), int(g[2]) if g[2] else int(g[1])]
                end   = [int(g[3]), int(g[4]), int(g[5]) if g[5] else int(g[4])]
                conv_info['kpoints']['range'] = [start, end]

        # ── ENCUT ─────────────────────────────────────────────────────────
        encut_match = re.search(r'ENCUT\s+(\d+)\s*[-–]\s*(\d+)', content, re.IGNORECASE)
        if encut_match:
            conv_info['encut']['enabled'] = True
            conv_info['encut']['range'] = [int(encut_match.group(1)),
                                           int(encut_match.group(2))]

        return conv_info
    
    def _extract_kpath(self, content: str):
        """Extract k-point path for band structure.
        Returns list of labels, or None if not specified (triggers auto-detection)."""
        kpath_match = re.search(r'along\s+([\w\-Γ]+)', content, re.IGNORECASE)
        if kpath_match:
            path_str = kpath_match.group(1).replace('Γ', 'G')
            labels = path_str.split('-')
            if labels:
                return labels
        return None  # auto-detect from structure
    
    def _extract_wannier(self, content: str) -> Dict[str, Any]:
        """Extract Wannier90 settings."""
        wannier_info = {
            'enabled':     'wannier' in content.lower(),
            'projections': [],
            'num_wann':    None,
            'num_bands':   None,
            'dis_win':     '',
        }

        if wannier_info['enabled']:
            # Projections from WANNIER_PROJ line or generic "projections: ..." line
            m = re.search(r'WANNIER_PROJ\s*[:,=]\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            if not m:
                m = re.search(r'project(?:ion)?s?\s*:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            if m:
                wannier_info['projections'] = [p.strip() for p in m.group(1).split(',')]

            m = re.search(r'WANNIER_NUM_WANN\s*[:,=]\s*(\d+)', content, re.IGNORECASE)
            if m:
                wannier_info['num_wann'] = int(m.group(1))

            m = re.search(r'WANNIER_EWIN\s*[:,=]\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            if m:
                wannier_info['dis_win'] = m.group(1).strip()

        return wannier_info
    
    def _extract_dfpt(self, content: str) -> dict:
        """Extract DFPT parameters."""
        return {
            'ediff': self._extract_float_key(content, r'DFPT_EDIFF\s*[:,=]\s*([-\d.Ee]+)',
                                              default=1e-8),
        }

    def _extract_phonons(self, content: str) -> dict:
        """Extract phonopy phonon calculation parameters."""
        dim_m  = re.search(r'PHONONS_DIM\s*[:,=]\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        band_m = re.search(r'PHONONS_BAND\s*[:,=]\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        mesh_m = re.search(r'PHONONS_MESH\s*[:,=]\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        nac_false = bool(re.search(r'PHONONS_NAC\s*[:,=]\s*(\.?FALSE\.?|no|0)',
                                   content, re.IGNORECASE))
        return {
            'dim':  dim_m.group(1).strip()  if dim_m  else '2 2 2',
            'band': band_m.group(1).strip() if band_m else '',
            'mesh': mesh_m.group(1).strip() if mesh_m else '20 20 20',
            'disp': self._extract_float_key(content, r'PHONONS_DISP\s*[:,=]\s*([-\d.Ee]+)',
                                            default=0.01),
            'nac':  not nac_false,
        }

    def _extract_kmesh(self, content: str):
        """Extract an explicit Gamma-mesh override for the automatic KPOINTS.

        Overrides the tiered density (low/medium/high/very_high) used by the
        generator's automatic mesh. Accepted forms (case-insensitive):

            KMESH: 8 8 8     KMESH = 8x8x8     KMESH: 12   (-> 12 12 12)

        Returns [nx, ny, nz] or None. For 2-D, give the third value
        explicitly (e.g. KMESH: 12 12 1).
        """
        m = re.search(r'KMESH\s*[:=]\s*(\d+(?:\s*[x×,\s]\s*\d+){0,2})',
                      content, re.IGNORECASE)
        if not m:
            return None
        nums = [int(n) for n in re.findall(r'\d+', m.group(1))]
        if not nums:
            return None
        if len(nums) == 1:
            return [nums[0], nums[0], nums[0]]
        if len(nums) == 2:
            return [nums[0], nums[1], nums[1]]
        return nums[:3]

    def _extract_pressure(self, content: str) -> dict:
        """Extract an externally imposed pressure for constant-pressure relaxation.

        Recognised forms (case-insensitive):
            PRESSURE = 10 GPa
            PRESSURE: 5 kbar
            pressure 10 GPa
            constant pressure 10 GPa
            apply 50 kbar pressure

        VASP's PSTRESS tag is in kBar (1 GPa = 10 kBar). If no unit is given
        the value is assumed to be in GPa (the convention most users quote),
        and that assumption is recorded so the INCAR comment can flag it.

        Returns {'enabled': bool, 'value': float, 'unit': 'GPa'|'kbar',
                 'pstress_kbar': float}.
        """
        info = {'enabled': False, 'value': None, 'unit': None, 'pstress_kbar': None}

        m = re.search(
            r'(?:constant\s+|apply\s+|external\s+)?'
            r'press(?:ure)?\s*[:=]?\s*([-+]?\d*\.?\d+)\s*(GPa|kbar|kB)?',
            content, re.IGNORECASE)
        if not m:
            # also catch "apply 50 kbar pressure" (value before the word)
            m = re.search(
                r'(?:apply|external|impose)\s+([-+]?\d*\.?\d+)\s*(GPa|kbar|kB)\s+press',
                content, re.IGNORECASE)
        if not m:
            return info

        value = float(m.group(1))
        unit  = (m.group(2) or '').lower()
        if unit in ('kbar', 'kb'):
            unit_label = 'kbar'
            pstress = value
        else:
            # default / explicit GPa
            unit_label = 'GPa'
            pstress = value * 10.0   # 1 GPa = 10 kBar

        info.update({'enabled': True, 'value': value,
                     'unit': unit_label, 'pstress_kbar': pstress})
        return info

    def _extract_incar_raw(self, content: str) -> dict:
        """Extract raw INCAR tag blocks written in normal VASP syntax.

        Lets the user drop literal INCAR lines into the instructions file so
        they end up verbatim in the generated INCAR(s). Two block forms are
        supported (case-insensitive header, terminated by END_INCAR / END INCAR):

            INCAR:                 # applies to every step
                LREAL = Auto
                ALGO  = Fast
                NELMIN = 6
            END_INCAR

            INCAR scf:             # applies only to the named step
                SIGMA = 0.1
            END_INCAR

        Recognised step names: relax, scf, bands, dos, wannier, dfpt, phonons
        (a block with no name, or named 'all', applies to all steps).

        Returns a dict mapping 'all'/<step> -> list of raw 'TAG = value' lines.
        Lines that don't look like INCAR assignments are ignored; inline
        comments (after the value) are preserved.
        """
        valid_steps = {'relax', 'scf', 'bands', 'dos', 'wannier', 'dfpt', 'phonons'}
        raw = {}

        block_re = re.compile(
            r'^[ \t]*INCAR(?:[ \t_]+([A-Za-z]+))?[ \t]*:[ \t]*\n'
            r'(.*?)'
            r'^[ \t]*END[ _]?INCAR[ \t]*$',
            re.IGNORECASE | re.DOTALL | re.MULTILINE)

        for m in block_re.finditer(content):
            name = (m.group(1) or 'all').lower()
            if name != 'all' and name not in valid_steps:
                # unknown qualifier → treat as global so the tags aren't lost
                name = 'all'
            tags = []
            for line in m.group(2).splitlines():
                s = line.strip()
                if not s or s.startswith('#') or s.startswith('!'):
                    continue
                if '=' not in s:
                    continue
                tags.append(s)
            if tags:
                raw.setdefault(name, []).extend(tags)

        return raw

    def _extract_transport(self, content: str) -> bool:
        """Check if transport calculations are requested"""
        return bool(re.search(r'(transport|boltzwann)', content, re.IGNORECASE))
    
    def _extract_dos_projections(self, content: str) -> List[str]:
        """Extract DOS projection specifications.

        Handles all common formats:
          Mo-d  Mo:d  Mo-2p  C-2s  S-p  Ni:3d
        Returns list of 'Element:orbital' strings, e.g. ['Mo:d', 'S:p', 'C:s']
        """
        projections = []
        proj_match = re.search(r'project(?:ed|ion[s]?)\s+on\s+(.+?)(?:\n|$)', content, re.IGNORECASE)
        if proj_match:
            proj_str = proj_match.group(1)
            # Match  Element[-:]optional_digit orbital_letter
            # e.g.  C-2s  Mo-d  S:p  Ni:3d
            matches = re.findall(r'([A-Z][a-z]?)[-:](\d*[spdf])', proj_str)
            projections = [f"{elem}:{orb[-1]}" for elem, orb in matches]
        return projections
    
    def get(self, key: str, default=None):
        """Get instruction value"""
        return self.instructions.get(key, default)
    
    def __repr__(self):
        """String representation"""
        return f"InstructionParser({self.instructions})"
