{

	gStyle->SetOptStat(0);
	TFile *f = new TFile("result.root","recreate");
		
	TNtuple *t = new TNtuple("t","","p1:p2:label");
		
	t->ReadFile("result");
	t->Draw("p2>>h0(200,0,1.1)","label==0");
	TH1F *h0 = (TH1F*)gDirectory->Get("h0");

	t->Draw("p2>>h1(200,0,1.1)","label==1");
	TH1F *h1 = (TH1F*)gDirectory->Get("h1");
	//h0->Scale(1./h0->Integral());
	//h1->Scale(1./h1->Integral());

	h0->GetYaxis()->SetTitle("a.u.");
	
	h0->GetYaxis()->SetRangeUser(1,2e4);

	h0->GetXaxis()->SetTitle("Prob_{phys}");
	h0->SetLineWidth(2);
	h1->SetLineColor(kRed);
	h1->SetLineWidth(2);
	h0->Draw("");
	c1->SetLogy();
	h1->Draw("same");

	TLegend *l = new TLegend(0.5,0.7,0.7,0.9);
	l->AddEntry(h0,"dark noise","l");
	l->AddEntry(h1,"physics","l");
	l->Draw();

	TLine *line = new TLine(0.3,0,0.3,2e4);
	line->SetLineColor(kGreen);
	line->SetLineWidth(2);
	line->Draw();

	t->Write();
	//f->Close();


}
