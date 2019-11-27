'''
Created on 17.11.2013

@author: thomas
'''

import numpy as N
from itertools import chain

def drophead(blocks,drop):
    '''
    Drop a given number of frames at the head of the stream
    '''
    assert drop >= 0
    blocks = iter(blocks)
    o = 0
    for b in blocks:
        o1 = o+len(b)
        if o1 < drop:
            pass # completely drop block
        else:
            b = b[drop-o:]
            if len(b):
                yield b
            break
        o = o1
        
    for b in blocks:
        yield b


def droptail(blocks,drop):
    '''
    Drop a given number of frames at the tail of the stream
    Strategy: retain enough frames in temporary memory and drop the superfluous frames in the end
    '''
    assert drop >= 0
    blocks = iter(blocks)
    mem = []
    lmem = 0
    for b in blocks:
        lb = len(b)
        mem.append(b)
        lmem += lb
        l0 = len(mem[0])
        if lmem-l0 >= drop:
            # memory length minus first block is still >= drop -> drop first block
            b = mem.pop(0)
            lmem -= l0
            yield b
            
    if len(mem):
        b = mem.pop(0)
        l0 = len(b)
        lmem -= l0
        assert lmem <= drop
        b1 = b[:lmem-drop or None]
        if len(b1):
            yield b1 
    else:
        assert drop == 0


def dropframes(blocks,fromhead=None,fromtail=None):
    if fromhead:
        blocks = drophead(blocks,fromhead)
    if fromtail:
        blocks = droptail(blocks,fromtail)
    return blocks


def reblock(blocks,blocksize,hopsize=None,dtype=None,fullsize=False,hastime=False,outtime=None):
    '''
    Reblock incoming stream of numpy arrays or lists containing numbers only
    '''
    
    assert blocksize > 0
    
    if hopsize is None:
        hopsize = blocksize
    else:
        assert hopsize > 0
        assert hopsize <= blocksize

    if hopsize*2 < blocksize:  # will hopping be multiply overlapped?
        # yes
        ovl = lambda x: x.copy()
    else:
        # no... don't need to copy input data (important case of hopsize = blocksize/2)
        ovl = lambda x: x 
        
    if outtime is None:
        outtime = hastime
    
    blocks = iter(blocks)
    
    bl0 = next(blocks)
    if hastime:
        _,b = bl0
    else:
        b = bl0
    b = N.asarray(b)  # take first block to know the shape
    res = N.empty((blocksize,)+b.shape[1:],dtype or b.dtype)

    if outtime:
        tms = N.empty(blocksize,dtype=float)
        
    ores = 0
    for b in chain((bl0,),blocks):
        if hastime:
            t,b = b
            assert len(t) == len(b)
            
        while len(b):
            take = min(blocksize-ores,len(b)) # number of frames to transfer
            res[ores:ores+take] = b[:take]  # copy frames from input to output
            if outtime:
                tms[ores:ores+take] = t[:take]  # copy frames from input to output
            ores += take  # add taken frames to output offset
            b = b[take:]  # set remainder of b after frames taken
            if outtime:
                t = t[take:]
            
            if ores == blocksize: # output buffer is full
                if outtime:
                    yield tms.copy(),res.copy() # we must copy, as the array will be updated later
                else:
                    yield res.copy() # we must copy, as the array will be updated later
                if hopsize < blocksize:
                    # copy parts of the buffer to account for hopsize
                    res[:-hopsize] = ovl(res[hopsize:])
                    if outtime:
                        tms[:-hopsize] = ovl(tms[hopsize:])
                ores -= hopsize
                
    # yield partially filled output buffer
    if ores:
        if fullsize:
            # zero out remainder of buffer
            res[ores:] = 0
            if outtime:
                tms[ores:] = N.inf  # difficult to judge what time should be filled in here
                yield tms,res                
            else:
                yield res
        else:
            if outtime:
                yield tms[:ores],res[:ores]
            else:
                yield res[:ores]


def izip_blks(*streams):
    '''
    Function works like izip, but returns blocks of equal size from the parallel input streams
    '''
    mem = [None for _ in streams]
    streams = [iter(s) for s in streams]
    while True:
        # get new stream elements if necessary
        mem = [(s.next() if l is None else l) for s,l in zip(streams,mem)] # StopIteration can be raised here
        # number of elements we can take from all the input blocks
        minlen = min(map(len,mem)) 
        ret = []
        mem1 = []
        for l in mem:
            ret.append(l[:minlen])
            l = l[minlen:] # remainder of block
            mem1.append(l if len(l) else None)
        mem = mem1
        yield tuple(ret)
                

    
# my old version - kind of inefficient
#
# def re_hop(blocks,blocksize,hopsize):
#     # block... dim 0 is temporal
#     res = []
#     lres = 0
#     offs = 0
#     for b in blocks:
#         res.append(b)
#         lres += len(b)
#         while lres-offs >= blocksize:
#             allres = N.concatenate(res) if len(res) > 1 else res[0]
#             yield allres[offs:offs+blocksize]
#             offs += hopsize
#             while len(res) and offs >= len(res[0]):
#                 bp = res.pop(0)
#                 lres -= len(bp)
#                 offs -= len(bp)
# 
#
# Jan's take on the problem:
# 
# def re_hop2(blocks, blocksize, hopsize):
#     """
#     Reads an iterable of blocks (multi-dimensional arrays, the first dimension
#     of which is the time dimension) of arbitrary size, but zero overlap, and
#     yields blocks of a new given blocksize and hopsize.
#     This version is much more efficient than re_hop() when input blocks are
#     shorter than output blocks.
#     @param blocks: Iterable of blocks
#     @param blocksize: Length of new blocks to form
#     @param hopsize: Time to skip between beginnings of blocks
#     """
#     blocks = iter(blocks)
#     b = blocks.next()
#     outblock = N.empty((blocksize,) + b.shape[1:], b.dtype)
#     outblock_filled = 0
#     try:
#         while True:
#             # fully consume input blocks and skip to next while we can
#             while outblock_filled + len(b) < blocksize:
#                 outblock[outblock_filled:outblock_filled + len(b)] = b
#                 outblock_filled += len(b)
#                 b = blocks.next()
#             # consume missing part of input block, save rest for the next time
#             if outblock_filled < blocksize:
#                 outblock[outblock_filled:] = b[:blocksize - outblock_filled]
#                 b = b[blocksize - outblock_filled:]  # (can become empty)
#                 outblock_filled = blocksize
#             # return output block
#             yield outblock
#             # hop on
#             if hopsize < blocksize:
#                 outblock[:-hopsize] = outblock[hopsize:].copy()
#                 outblock_filled = blocksize - hopsize
#             else:
#                 # skip hopsize - blocksize input frames
#                 outblock_filled = 0
#                 skip = hopsize - blocksize - len(b)
#                 while skip > 0:
#                     b = blocks.next()
#                     skip -= len(b)
#                 if skip < 0:
#                     # we skipped too far. keep the last part of b.
#                     b = b[skip:]
#     except StopIteration:
#         pass

