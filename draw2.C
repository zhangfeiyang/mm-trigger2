{
	ifstream fin("Data5_15");
	
	const int n = 42489;	
	
	int data[325];
	int label;
	
	TFile *f = new TFile("Data5_15.root","recreate");
	TTree *t = new TTree("t","");
	t->Branch("data",data,"data[325]/I");
	t->Branch("label",&label,"label/I");
	
	TH1F *h = new TH1F("h","",100,0,100);
	
		
	for(int i=0;i<4;i++){
		
		for(int j=0;j<325;j++){	
			fin>>data[j];
			cout << data[j] << "\n";
			if(i==0 && j>=225)
				h->SetBinContent(j+1-225,data[j]);	
		}
	
		fin>>label;
		t->Fill();	
	} 
	
	h->Draw();
	
	//t->Write();
	
	//f->Close();

}
