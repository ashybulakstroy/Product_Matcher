"""Synthetic tests; stdlib unittest, no production data or OCR photos needed."""
import unittest
from App.invoicing.matching_bridge import confident_default, stock_rows, vote_candidates, fallback_suggestions


CATALOG={'items':[
    {'id':'00000123','code':'00000123','name':'Кабель медный 2 мм','stock_quantity':10,'accounting_price':'100'},
    {'id':'00000234','code':'00000234','name':'Кабель медный 4 мм','stock_quantity':8,'accounting_price':'120'},
]}


def matcher(checked,stock,**kwargs):
    assert checked[0]['name']=='Кабель медный'
    assert kwargs['min_stock']==2
    return {'results_by_algorithm':[], 'voting': {'positions':[{
        'voting_algorithms':3,'is_tie':False,'recommended_stock_id':'00000123',
        'candidates':[{'stock_id':'00000123','votes':2,'vote_pct':66.67,'voters':['a','b']},
                      {'stock_id':'00000234','votes':1,'vote_pct':33.33,'voters':['c']}]}]}}


class MatchingBridgeTests(unittest.TestCase):
    def test_confidence_threshold_and_margin(self):
        self.assertTrue(confident_default([{'score':92},{'score':75}]))
        self.assertFalse(confident_default([{'score':84}]))
        self.assertFalse(confident_default([{'score':92},{'score':90}]))

    def test_voting_preserves_leading_zeroes(self):
        data=vote_candidates('Кабель медный', CATALOG, quantity=2, matcher=matcher)
        self.assertEqual(data['recommended_id'],'00000123')
        self.assertEqual(data['items'][0]['code'],'00000123')
        self.assertEqual(data['items'][0]['votes'],2)

    def test_fallback_only_for_uncertain(self):
        self.assertEqual(fallback_suggestions({'name':'Кабель медный','quantity':2},
                          [{'id':'00000123','score':95}],CATALOG,matcher=matcher),[])
        result=fallback_suggestions({'name':'Кабель медный','quantity':2},
                          [{'id':'00000123','score':72}],CATALOG,matcher=matcher)
        self.assertEqual([r['code'] for r in result],['00000234'])

    def test_ambiguous_identity_not_assigned(self):
        _, by_id = stock_rows({'items':[CATALOG['items'][0],CATALOG['items'][0]]})
        self.assertNotIn('00000123',by_id)

    def test_tied_winner_never_selected(self):
        def tied(*args, **kwargs):
            return {'voting':{'positions':[{'is_tie':True,'recommended_stock_id':None,
                'candidates':[{'stock_id':'00000123','votes':1},{'stock_id':'00000234','votes':1}]}]},
                'results_by_algorithm':[]}
        data=vote_candidates('Кабель медный',CATALOG,quantity=2,matcher=tied)
        self.assertIsNone(data['recommended_id'])


if __name__=='__main__':
    unittest.main()
