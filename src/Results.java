
public class Results {
	double TP;
	double FP;
	double TN;
	double FN;
	double precision;
	double recall;
	double pf;
	double fScore;
	double balance;
	public Results(double tP, double fP, double tN, double fN, double precision, double recall, double pf, double fScore,
			double balance) {
		super();
		TP = tP;
		FP = fP;
		TN = tN;
		FN = fN;
		this.precision = precision;
		this.recall = recall;
		this.pf = pf;
		this.fScore = fScore;
		this.balance = balance;
	}
	public double getTP() {
		return TP;
	}
	public void setTP(double tP) {
		TP = tP;
	}
	public double getFP() {
		return FP;
	}
	public void setFP(double fP) {
		FP = fP;
	}
	public double getTN() {
		return TN;
	}
	public void setTN(double tN) {
		TN = tN;
	}
	public double getFN() {
		return FN;
	}
	public void setFN(double fN) {
		FN = fN;
	}
	public double getPrecision() {
		return precision;
	}
	public void setPrecision(double precision) {
		this.precision = precision;
	}
	public double getRecall() {
		return recall;
	}
	public void setRecall(double recall) {
		this.recall = recall;
	}
	public double getPf() {
		return pf;
	}
	public void setPf(double pf) {
		this.pf = pf;
	}
	public double getfScore() {
		return fScore;
	}
	public void setfScore(double fScore) {
		this.fScore = fScore;
	}
	public double getBalance() {
		return balance;
	}
	public void setBalance(double balance) {
		this.balance = balance;
	}

	
}
