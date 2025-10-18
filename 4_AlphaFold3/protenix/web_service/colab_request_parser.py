# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import subprocess
import traceback
from collections import defaultdict
from copy import deepcopy
from os.path import exists as opexists
from os.path import join as opjoin
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import protenix.data.ccd as ccd
import requests
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.web_service.colab_request_utils import run_mmseqs2_service
from protenix.web_service.dependency_url import URL

MMSEQS_SERVICE_HOST_URL = os.getenv(
    "MMSEQS_SERVICE_HOST_URL", "https://protenix-server.com/api/msa"
)
MAX_ATOM_NUM = 60000
MAX_TOKEN_NUM = 5000
DATA_CACHE_DIR = "/af3-dev/release_data/"
CHECKPOINT_DIR = "/af3-dev/release_model/"


def download_tos_url(tos_url, local_file_path):
    try:
        response = requests.get(tos_url, stream=True)

        if response.status_code == 200:
            with open(local_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Succeeded downloading from {tos_url}.\nSaved to {local_file_path}.")
        else:
            print(
                f"Failed downloading from {tos_url}.\nStatus code: {response.status_code}"
            )

    except Exception as e:
        print(f"Error occured in downloading: {e}")


class TooLargeComplexError(Exception):
    def __init__(self, **kwargs) -> None:
        if "num_atoms" in kwargs:
            message = (
                f"We can only process complexes with no more than {MAX_ATOM_NUM} atoms, "
                f"but there are {kwargs['num_atoms']} atoms in the input."
            )
        elif "num_tokens" in kwargs:

            message = (
                f"We can only process complexes with no more than {MAX_TOKEN_NUM} tokens, "
                f"but there are {kwargs['num_tokens']} tokens in the input."
            )
        else:
            message = ""
        super().__init__(message)


class RequestParser(object):
    def __init__(
        self, request_json_path: str, request_dir: str, email: str = ""
    ) -> None:
        with open(request_json_path, "r") as f:
            self.request = json.load(f)
        self.request_dir = request_dir
        self.fpath = os.path.abspath(__file__)
        self.email = email
        os.makedirs(self.request_dir, exist_ok=True)

    def download_data_cache(self) -> Dict[str, str]:
        data_cache_dir = DATA_CACHE_DIR
        os.makedirs(data_cache_dir, exist_ok=True)
        cache_paths = {}
        for cache_name, fname in [
            ("ccd_components_file", "components.v20240608.cif"),
            ("ccd_components_rdkit_mol_file", "components.v20240608.cif.rdkit_mol.pkl"),
        ]:
            if not opexists(
                cache_path := os.path.abspath(opjoin(data_cache_dir, fname))
            ):
                tos_url = URL[cache_name]
                print(f"Downloading data cache from\n {tos_url}...")
                download_tos_url(tos_url, cache_path)
            cache_paths[cache_name] = cache_path
        return cache_paths

    def download_model(self, model_version: str, checkpoint_local_path: str) -> None:
        tos_url = URL[f"model_{model_version}"]
        print(f"Downloading model checkpoing from\n {tos_url}...")
        download_tos_url(tos_url, checkpoint_local_path)

    def get_model(self) -> str:
        checkpoint_dir = CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_version = self.request["model_version"]
        if not opexists(
            checkpoint_path := opjoin(checkpoint_dir, f"model_{model_version}.pt")
        ):
            self.download_model(model_version, checkpoint_local_path=checkpoint_path)
        if opexists(checkpoint_path):
            return checkpoint_path
        else:
            raise ValueError("Failed in finding model checkpoint.")

    def get_data_json(self) -> str:
        input_json_dict = {
            "name": (self.request["name"]),
            "covalent_bonds": self.request["covalent_bonds"],
        }
        input_json_path = opjoin(self.request_dir, f"inputs.json")

        sequences = []
        entity_pending_msa = {}
        for i, entity_info_wrapper in enumerate(self.request["sequences"]):
            entity_id = str(i + 1)
            entity_info_wrapper: Dict[str, Dict[Any]]
            assert len(entity_info_wrapper) == 1

            seq_type, seq_info = next(iter(entity_info_wrapper.items()))

            if seq_type == "proteinChain":
                if self.request["use_msa"]:
                    entity_pending_msa[entity_id] = seq_info["sequence"]

            if seq_type not in [
                "proteinChain",
                "dnaSequence",
                "rnaSequence",
                "ligand",
                "ion",
            ]:
                raise NotImplementedError
            sequences.append({seq_type: seq_info})

        tmp_json_dict = deepcopy(input_json_dict)
        tmp_json_dict["sequences"] = sequences

        cache_paths = self.download_data_cache()
        ccd.COMPONENTS_FILE = cache_paths["ccd_components_file"]
        ccd.RKDIT_MOL_PKL = Path(cache_paths["ccd_components_rdkit_mol_file"])
        sample2feat = SampleDictToFeatures(
            tmp_json_dict,
        )
        atom_array = sample2feat.get_atom_array()
        num_atoms = len(atom_array)
        num_tokens = np.sum(atom_array.centre_atom_mask)
        if num_atoms > MAX_ATOM_NUM:
            raise TooLargeComplexError(num_atoms=num_atoms)
        if num_tokens > MAX_TOKEN_NUM:
            raise TooLargeComplexError(num_tokens=num_tokens)
        del tmp_json_dict

        if len(entity_pending_msa) > 0:
            seq_to_entity_id = defaultdict(list)
            for entity_id, seq in entity_pending_msa.items():
                seq_to_entity_id[seq].append(entity_id)
            seq_to_entity_id = dict(seq_to_entity_id)
            seqs_pending_msa = sorted(list(seq_to_entity_id.keys()))

            os.makedirs(msa_res_dir := opjoin(self.request_dir, "msa"), exist_ok=True)

            tmp_fasta_fpath = opjoin(msa_res_dir, "msa_input.fasta")
            RequestParser.msa_search(
                seqs_pending_msa=seqs_pending_msa,
                tmp_fasta_fpath=tmp_fasta_fpath,
                msa_res_dir=msa_res_dir,
                email=self.email,
            )
            msa_res_subdirs = RequestParser.msa_postprocess(
                seqs_pending_msa=seqs_pending_msa,
                msa_res_dir=msa_res_dir,
            )

            for seq, msa_res_dir in zip(seqs_pending_msa, msa_res_subdirs):
                for entity_id in seq_to_entity_id[seq]:
                    entity_index = int(entity_id) - 1
                    sequences[entity_index]["proteinChain"]["msa"] = {
                        "precomputed_msa_dir": msa_res_dir,
                        "pairing_db": "uniref100",
                        "pairing_db_fpath": None,
                        "non_pairing_db_fpath": None,
                        "search_too": None,
                        "msa_save_dir": None,
                    }

        input_json_dict["sequences"] = sequences
        with open(input_json_path, "w") as f:
            json.dump([input_json_dict], f, indent=4)
        return input_json_path

    @staticmethod
    def msa_search(
        seqs_pending_msa: Sequence[str],
        tmp_fasta_fpath: str,
        msa_res_dir: str,
        email: str = "",
    ) -> None:
        lines = []
        for idx, seq in enumerate(seqs_pending_msa):
            lines.append(f">query_{idx}\n")
            lines.append(f"{seq}\n")
        if (last_line := lines[-1]).endswith("\n"):
            lines[-1] = last_line.rstrip("\n")
        with open(tmp_fasta_fpath, "w") as f:
            for lines in lines:
                f.write(lines)

        with open(tmp_fasta_fpath, "r") as f:
            query_seqs = f.read()
        try:
            run_mmseqs2_service(
                query_seqs,
                msa_res_dir,
                True,
                use_templates=False,
                host_url=MMSEQS_SERVICE_HOST_URL,
                user_agent="colabfold/1.5.5",
                email=email,
            )
        except Exception as e:
            error_message = f"MMSEQS2 failed with the following error message:\n{traceback.format_exc()}"
            print(error_message)

    @staticmethod
    def msa_postprocess(seqs_pending_msa: Sequence[str], msa_res_dir: str) -> None:
        def read_m8(m8_file: str) -> Dict[str, str]:
            uniref_to_ncbi_taxid = {}
            with open(m8_file, "r") as infile:
                for line in infile:
                    line_list = line.replace("\n", "").split("\t")
                    hit_name = line_list[1]
                    ncbi_taxid = line_list[2]
                    uniref_to_ncbi_taxid[hit_name] = ncbi_taxid
            return uniref_to_ncbi_taxid

        def read_a3m(a3m_file: str) -> Tuple[List[str], List[str]]:
            heads = []
            seqs = []
            # Record the row index. The index before this index is the MSA of Uniref30 DB,
            # and the index after this index is the MSA of ColabfoldDB.
            uniref_index = 0
            with open(a3m_file, "r") as infile:
                for idx, line in enumerate(infile):
                    if line.startswith(">"):
                        heads.append(line)
                        if idx == 0:
                            query_name = line
                        elif idx > 0 and line == query_name:
                            uniref_index = idx
                    else:
                        seqs.append(line)
            return heads, seqs, uniref_index

        def make_pairing_and_non_pairing_msa(
            query_seq: str,
            seq_dir: str,
            raw_a3m_path: str,
            uniref_to_ncbi_taxid: Mapping[str, str],
        ) -> List[str]:

            heads, msa_seqs, uniref_index = read_a3m(raw_a3m_path)
            uniref100_lines = [">query\n", f"{query_seq}\n"]
            other_lines = [">query\n", f"{query_seq}\n"]

            for idx, (head, msa_seq) in enumerate(zip(heads, msa_seqs)):
                if msa_seq.rstrip("\n") == query_seq:
                    continue

                uniref_id = head.split("\t")[0][1:]
                ncbi_taxid = uniref_to_ncbi_taxid.get(uniref_id, None)
                if (ncbi_taxid is not None) and (idx < (uniref_index // 2)):
                    if not uniref_id.startswith("UniRef100_"):
                        head = head.replace(
                            uniref_id, f"UniRef100_{uniref_id}_{ncbi_taxid}/"
                        )
                    else:
                        head = head.replace(uniref_id, f"{uniref_id}_{ncbi_taxid}/")
                    uniref100_lines.extend([head, msa_seq])
                else:
                    other_lines.extend([head, msa_seq])

            with open(opjoin(seq_dir, "pairing.a3m"), "w") as f:
                for line in uniref100_lines:
                    f.write(line)
            with open(opjoin(seq_dir, "non_pairing.a3m"), "w") as f:
                for line in other_lines:
                    f.write(line)

        def make_non_pairing_msa_only(
            query_seq: str,
            seq_dir: str,
            raw_a3m_path: str,
        ):
            heads, msa_seqs, _ = read_a3m(raw_a3m_path)
            other_lines = [">query\n", f"{query_seq}\n"]
            for head, msa_seq in zip(heads, msa_seqs):
                if msa_seq.rstrip("\n") == query_seq:
                    continue
                other_lines.extend([head, msa_seq])
            with open(opjoin(seq_dir, "non_pairing.a3m"), "w") as f:
                for line in other_lines:
                    f.write(line)

        def make_dummy_msa(
            query_seq: str, seq_dir: str, msa_type: str = "both"
        ) -> None:
            if msa_type == "both":
                fnames = ["pairing.a3m", "non_pairing.a3m"]
            elif msa_type == "pairing":
                fnames = ["pairing.a3m"]
            elif msa_type == "non_pairing":
                fnames = ["non_pairing.a3m"]
            else:
                raise NotImplementedError
            for fname in fnames:
                with open(opjoin(seq_dir, fname), "w") as f:
                    f.write(">query\n")
                    f.write(f"{query_seq}\n")

        msa_res_subdirs = []
        for seq_idx, query_seq in enumerate(seqs_pending_msa):
            os.makedirs(
                seq_dir := os.path.abspath(opjoin(msa_res_dir, str(seq_idx))),
                exist_ok=True,
            )
            if opexists(raw_a3m_path := opjoin(msa_res_dir, f"{seq_idx}.a3m")):
                if opexists(m8_path := opjoin(msa_res_dir, "uniref_tax.m8")):
                    uniref_to_ncbi_taxid = read_m8(m8_path)
                    make_pairing_and_non_pairing_msa(
                        query_seq=query_seq,
                        seq_dir=seq_dir,
                        raw_a3m_path=raw_a3m_path,
                        uniref_to_ncbi_taxid=uniref_to_ncbi_taxid,
                    )
                else:
                    make_non_pairing_msa_only(
                        query_seq=query_seq,
                        seq_dir=seq_dir,
                        raw_a3m_path=raw_a3m_path,
                    )
                    make_dummy_msa(
                        query_seq=query_seq, seq_dir=seq_dir, msa_type="pairing"
                    )

            else:
                print(
                    f"Failed in searching MSA for \n{query_seq}\nusing the sequence itself as MSA."
                )
                make_dummy_msa(query_seq=query_seq, seq_dir=seq_dir)
            msa_res_subdirs.append(seq_dir)

        return msa_res_subdirs

    def launch(self) -> None:
        input_json_path = self.get_data_json()
        checkpoint_path = self.get_model()

        entry_path = os.path.abspath(
            opjoin(os.path.dirname(self.fpath), "../../runner/inference.py")
        )
        command_parts = [
            "export LAYERNORM_TYPE=fast_layernorm;",
            f"python3 {entry_path}",
            f"--load_checkpoint_path {checkpoint_path}",
            f"--dump_dir {self.request_dir}",
            f"--input_json_path {input_json_path}",
            f"--need_atom_confidence {self.request['atom_confidence']}",
            f"--use_msa {self.request['use_msa']}",
            "--num_workers 0",
            "--dtype bf16",
            "--use_deepspeed_evo_attention True",
            "--sample_diffusion.step_scale_eta 1.5",
        ]

        if "model_seeds" in self.request:
            seeds = ",".join([str(seed) for seed in self.request["model_seeds"]])
            command_parts.extend([f'--seeds "{seeds}"'])
        for key in ["N_sample", "N_step"]:
            if key in self.request:
                command_parts.extend([f"--sample_diffusion.{key} {self.request[key]}"])
        if "N_cycle" in self.request:
            command_parts.extend([f"--model.N_cycle {self.request['N_cycle']}"])
        command = " ".join(command_parts)
        print(f"Launching inference process with the command below:\n{command}")
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--request_json_path",
        type=str,
        required=True,
        help="Path to the request JSON file.",
    )
    parser.add_argument(
        "--request_dir", type=str, required=True, help="Path to the request directory."
    )
    parser.add_argument(
        "--email", type=str, required=False, default="", help="Your email address."
    )

    args = parser.parse_args()
    parser = RequestParser(
        request_json_path=args.request_json_path,
        request_dir=args.request_dir,
        email=args.email,
    )
    parser.launch()
