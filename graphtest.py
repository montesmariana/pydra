import sys
import os
sys.path.append('./')
import pydra

@pydra.mark.task
def toupper(corpus):
  return 'corpus-' + corpus.upper()

@pydra.mark.task
def repeat(corpus, windowsize):
  return f"{corpus} with {windowsize}-{windowsize} windowspan"


if __name__ == '__main__':
    cache_dir = os.path.join(os.getcwd(), 'cache')
    inputs = {
      'corpus' : ['a', 'b', 'c'],
      'windowsize' : [1,2, 3]
    }
    wf = pydra.Workflow(
        name = 'wf',
        input_spec = ['corpus', 'windowsize'],
        **inputs,
        cache_dir = cache_dir
    )
    wf.split(['corpus', 'windowsize'])
    wf.add(toupper(name = 'upper', corpus = wf.lzin.corpus))
    wf.add(repeat(name = 'rep', corpus = wf.upper.lzout.out, windowsize = wf.lzin.windowsize))
    wf.set_output([('out', wf.rep.lzout.out)])
    with pydra.Submitter(plugin = 'cf') as sub:
        sub(runnable = wf)
    result = wf.result(return_inputs = True)
    print(result)
    file = wf.create_dotfile(type = 'detailed', name = 'wf-detailed', output_dir = './')
    print(file)

