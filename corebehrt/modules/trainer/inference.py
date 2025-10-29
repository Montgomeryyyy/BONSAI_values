from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch


class EHRInferenceRunner(EHRTrainer):
    def inference_loop(self, return_embeddings=False) -> tuple:
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(
            "Running inference with embeddings"
            if return_embeddings
            else "Running inference"
        )

        logits, targets = [], []
        if return_embeddings:
            self.model.cls.eval()
            model_embs, head_embs, att_masks = [], [], []

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                if return_embeddings:
                    model = outputs.last_hidden_state
                    att = batch["attention_mask"]
                    head = self.extract_head_embeddings(batch, outputs)
                    model_embs.append(model.cpu())
                    head_embs.append(head.cpu())
                    att_masks.append(att.cpu())

                logits.append(outputs.logits.float().cpu())
                targets.append(batch["target"].cpu())

        logits_tensor = torch.cat(logits, dim=0).squeeze()
        targets_tensor = torch.cat(targets, dim=0).squeeze()

        embeddings = (
            [
                torch.cat(model_embs, dim=0).squeeze(),
                torch.cat(head_embs, dim=0).squeeze(),
                torch.cat(att_masks, dim=0).squeeze(),
            ]
            if return_embeddings
            else None
        )

        return logits_tensor, targets_tensor, embeddings

    def extract_head_embeddings(self, batch, outputs):
        head_embedding = self.model.cls(
            outputs.last_hidden_state,
            attention_mask=batch["attention_mask"],
            return_embedding=True,
        )
        return head_embedding


class EHRInferenceRunnerPretrain(EHRTrainer):
    def inference_loop(self, return_embeddings=False) -> tuple:
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        loop = get_tqdm(dataloader)
        loop.set_description(
            "Running inference with embeddings"
            if return_embeddings
            else "Running inference"
        )

        # Collect all embeddings on GPU first, then move to CPU once at the end
        model_embs = []
        batch_pids = []  # Store PIDs for each batch

        with torch.no_grad():
            for batch_idx, batch in enumerate(loop):
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                # Keep on GPU to avoid repeated CPU transfers
                model_embs.append(outputs.last_hidden_state)
                
                # Extract PIDs for this batch
                if hasattr(self.test_dataset, 'patients'):
                    # Get the indices for this batch
                    start_idx = batch_idx * self.args.get("test_batch_size", 128)
                    end_idx = min(start_idx + self.args.get("test_batch_size", 128), len(self.test_dataset))
                    batch_indices = list(range(start_idx, end_idx))
                    
                    # Get PIDs for these indices
                    pids = [self.test_dataset.patients[i].pid for i in batch_indices]
                    batch_pids.append(pids)
                else:
                    # Fallback: use batch indices as PIDs
                    batch_size = outputs.last_hidden_state.shape[0]
                    pids = [f"batch_{batch_idx}_sample_{i}" for i in range(batch_size)]
                    batch_pids.append(pids)
                
                # Optional: Clear GPU cache periodically to prevent OOM
                if batch_idx % 10 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Flatten embeddings and PIDs into single lists
        # Each element in the lists corresponds to one sample:
        # - embeddings: shape (sequence_length, hidden_size) - variable sequence length
        # - pids: string - patient ID
        
        # Flatten embeddings and PIDs
        all_embeddings = []
        all_pids = []
        
        for i, (emb, pids) in enumerate(zip(model_embs, batch_pids)):
            # Move to CPU and split by sample
            emb_cpu = emb.cpu()
            for j in range(emb_cpu.shape[0]):  # For each sample in the batch
                all_embeddings.append(emb_cpu[j])  # Shape: (sequence_length, hidden_size)
                all_pids.append(pids[j])  # String: patient ID
            
            # Clear GPU cache periodically
            if i % 10 == 0 and self.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Clear GPU memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        return all_embeddings, all_pids
