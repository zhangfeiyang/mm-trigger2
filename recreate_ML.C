{

	int Index[43857];
	ifstream fin("index");
		
	for(int i=0;i<43857;i++){
		fin>>Index[i];
	}

	TFile *f = new TFile("ML.root","read"); 
		
	TTree* t = (TTree*)f->Get("t");
	int pmtID[20000];
	int nhits,nhits_C14;

	t->SetBranchAddress("pmtID",pmtID);
	t->SetBranchAddress("nhits",&nhits);
	t->SetBranchAddress("nhits_C14",&nhits_C14);
	int entries = t->GetEntries();
	TFile *f0 = new TFile("ML2.root","recreate");
		
	TTree* t0 = new TTree("t","");
        t0->Branch("nhits",&nhits,"nhits/I");
    	t0->Branch("nhits_C14",&nhits_C14,"nhits_C14/I");
        t0->Branch("pmtID",pmtID,"pmtID[nhits]/I");

	for(int i=0;i<entries;i++){
		t->GetEntry(Index[i]);
		t0->Fill();
	}
	t0->Write();
	f0->Close();

}
