import os
import yaml
import numpy as np
import natumpy as nt
from rich.console import Console

console = Console()

class BrainManager:
    def __init__(self, config_path="brain.yaml"):
        self.config = self._load_config(config_path)
        self.name = self.config['name']
        self.dim = self.config['core']['dim']
        self.data_path = self.config['storage']['path']
        
        os.makedirs(self.data_path, exist_ok=True)
        self.brain_file = os.path.join(self.data_path, f"{self.name}.nawa")
        self.readout_file = os.path.join(self.data_path, f"{self.name}_readout.npy")
        
        self.engine = None
        self.tokenizer = None
        self.readout = None

    def _load_config(self, path):
        if not os.path.exists(path):
            console.print(f"[red] Config {path} tidak ditemukan![/red]")
            exit(1)
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def initialize(self):
        console.print(f"[bold cyan] Menginisialisasi {self.name} (Dim: {self.dim})...[/bold cyan]")
        
        self.tokenizer = nt.text.ResonantTokenizer(self.dim)
        
        if os.path.exists(self.brain_file):
            console.print("    Memuat struktur otak dari disk...")
            self.engine = nt.layers.ReservoirLayer(self.dim)
            self.engine.load(self.brain_file)
        else:
            console.print("    Menciptakan struktur otak baru...")
            self.engine = nt.layers.ReservoirLayer(self.dim)
            phy = self.config['physics']
            self.engine.set_config(phy['beta'], phy['dt'], phy['stiffness'], phy['decay'])

        self.readout = nt.layers.ReadoutLayer(self.dim * 4, self.dim * 2, alpha=self.config['training']['alpha'])
        
        if os.path.exists(self.readout_file):
            console.print("    Memuat keahlian bahasa...")
            self.readout.load(self.readout_file)

    def train(self, data_folder):
        import glob
        files = glob.glob(os.path.join(data_folder, "*.txt"))
        
        if not files:
            console.print("[yellow]⚠ Tidak ada file .txt di folder data![/yellow]")
            return

        all_text = ""
        for f in files:
            console.print(f"    Membaca: {os.path.basename(f)}")
            with open(f, 'r', encoding='utf-8') as txt:
                all_text += txt.read() + " "
        
        all_text = all_text.replace("\n", " ").strip()
        console.print(f"   [green]Total Karakter: {len(all_text)}[/green]")

        inputs_r, inputs_i = self.tokenizer.encode(all_text)
        targets_r, targets_i = self.tokenizer.encode(all_text[1:] + " ")
        
        X_states = []
        Y_targets = []
        
        console.print("    Resonansi Gelombang...")
        steps = len(inputs_r)
        for t in range(steps):
            state = self.engine.forward(inputs_r[t], inputs_i[t])
            X_states.append(state)
            
            target = np.concatenate([targets_r[t], targets_i[t]])
            Y_targets.append(target)
            
        console.print("   🎓 Melatih Saraf Bahasa (Linear Algebra)...")
        self.readout.fit(np.array(X_states), np.array(Y_targets))
        
        self.save()
        console.print("[bold green] Pelatihan Selesai & Tersimpan![/bold green]")

    def chat(self):
        if self.readout.W_out is None:
            console.print("[red] Model belum dilatih! Jalankan 'manest train' dulu.[/red]")
            return

        console.print(f"[bold yellow] Memulai Sesi Obrolan dengan {self.name}[/bold yellow]")
        console.print("(Ketik 'exit' untuk keluar)\n")
        
        while True:
            try:
                user_in = console.input("[bold green]User:[/bold green] ")
                if user_in.lower() in ['exit', 'keluar']: break
                
                prompt = f"User: {user_in} {self.name}:"
                
                pr, pi = self.tokenizer.encode(prompt)
                for i in range(len(pr)):
                    self.engine.forward(pr[i], pi[i])
                
                console.print(f"[bold cyan]{self.name}:[/bold cyan] ", end="")
                
                response = ""
                for _ in range(200):
                    state = self.engine.get_state()
                    pred_vec = self.readout.predict(state)
                    
                    pr_r = pred_vec[:self.dim]
                    pr_i = pred_vec[self.dim:]
                    
                    char = self.tokenizer.decode_sequence([pr_r], [pr_i])
                    idx = self.tokenizer.decode(pr_r, pr_i)
                    
                    response += char
                    console.print(char, end="")
                    
                    if "User:" in response or "\n" in char:
                        break
                        
                    nr = self.tokenizer.embeddings_r[idx]
                    ni = self.tokenizer.embeddings_i[idx]
                    self.engine.forward(nr, ni)
                
                print()
                
            except KeyboardInterrupt:
                break

    def save(self):
        self.engine.save(self.brain_file)
        self.readout.save(self.readout_file)
_file)
