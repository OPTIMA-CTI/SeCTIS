# 🛡️ SeCTIS: Secure CTI Sharing Framework

SeCTIS is a novel framework designed to enable **privacy-preserving**, **trustworthy**, and **automated** sharing of Cyber Threat Intelligence (CTI) across organizations using advanced technologies like **Swarm Learning**, **Blockchain**, and **Zero-Knowledge Proofs (ZKPs)**.


## 🧠 How It Works

1. **Local Model Training**: Organizations train models on their private CTI data.
2. **Model Update Sharing**: Only the model parameters are shared through the Swarm Network.
3. **Validator Verification**: Random validator nodes assess model quality and behavior using ZKPs.
4. **Reputation Calculation**: A Swarm Aggregator computes reputation scores and aggregates the top-k local models.
5. **Global Model Deployment**: The global model is updated in IPFS and distributed for the next iteration.

## 📊 Key Contributions

- 🔐 **Privacy-Preserving CTI Sharing**: No raw data is shared; only model parameters.
- 🤝 **Trust & Quality Assessment**: Reputation scores and ZKPs ensure data and model integrity.
- 🔄 **Interoperability & Automation**: Middleware supports various data formats and ML frameworks.
- 📈 **Scalability**: Distributed training reduces central bottlenecks.
- ⚖️ **Legal Compliance**: Minimizes risk via decentralized sharing and proof mechanisms.

## 📁 Repository Structure

├── src/ # Source code for experiments and SeCTIS components

  ├── main.py # Main script to run the SeCTIS training pipeline
  
## ⚙️ How to Run

Make sure you have Python 3.x installed and necessary packages.

pip install -r requirements.txt
python main.py

## 📚 Citation

If you use this in your research, please cite the following paper:

```bibtex
@article{arikkat2025sectis,
  title={Sectis: A framework to secure cti sharing},
  author={Arikkat, Dincy R and Cihangiroglu, Mert and Conti, Mauro and KA, Rafidha Rehiman and Nicolazzo, Serena and Nocera, Antonino and others},
  journal={Future Generation Computer Systems},
  volume={164},
  pages={107562},
  year={2025},
  publisher={Elsevier}
}
