# https://en.wikipedia.org/wiki/Greek_alphabet
greek_upr=['Α','Β','Γ','Δ','Ε','Ζ','Η','Θ','Ι','Κ','Λ','Μ','Ν','Ξ','Ο','Π','Ρ','Σ','Τ','Υ','Φ','Χ','Ψ','Ω'];
greek_lwr=['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω'];
greek_xtra_mapping={'ς':'Σ','ϲ':'Σ'};
accents=['','\u0300','\u0301','\u0304','\u0306','\u0308','\u0313','\u0314','\u0342','\u0343','\u0344','\u0345'];
accent_free=[''];

def _get_greek_v1(accent):
    chrs_upr = [];
    chrs_lwr = [];
    masters = [];
    servants = [];
    for cid in range(len(greek_upr)):
        for a in accent:
            masters.append(greek_upr[cid] + a);
            chrs_upr.append(greek_upr[cid]);
            servants.append(greek_lwr[cid] + a);
            chrs_lwr.append(greek_lwr[cid]);
    for s in greek_xtra_mapping:
        for a in accent:
            chrs_lwr.append(s + a);
            servants.append(s + a);
            masters.append(greek_xtra_mapping[s] + a);
    return chrs_upr, chrs_lwr, masters, servants;
def get_accented_greek_v1():
    return _get_greek_v1(accents);
def get_accented_free_greek_v1():
    return _get_greek_v1(accent_free);

if __name__ == '__main__':
    for l in get_accented_free_greek_v1():
        print(l)