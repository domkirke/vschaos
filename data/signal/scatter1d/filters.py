import numpy as np

__all__ = ("filters", "filter_ds")

def morlet_freq_1d(filt_opt):
    """
    Compute center frequencies and bandwidths for the 1D Morlet
    
    Usage
       [psi_xi, psi_bw, phi_bw] = morlet_freq_1d(filt_opt)
    
    Input
       filt_opt (struct): The parameters defining the filter bank. See 
          'morlet_freq_bank_1d' for details.
    
    Output
       psi_xi (numeric): The center frequencies of the wavelet filters.
       psi_bw (numeric): The bandwidths of the wavelet filters.
       phi_bw (numeric): The bandwidth of the lowpass filter.
    
    Description
       Compute the center frequencies and bandwidth for the wavelets and lowpass
       filter of the one-dimensional Morlet/Gabor filter bank.
    
    See also
       FILTER_FREQ, DYADIC_FREQ_1D
    """ 

#     sigma0 = 2/sqrt(3);
    sigma0 = 2./np.sqrt(3)
    
    # Calculate logarithmically spaced, band-pass filters.
#     xi_psi = filt_opt.xi_psi * 2.^((0:-1:1-filt_opt.J)/filt_opt.Q);
#     sigma_psi = filt_opt.sigma_psi * 2.^((0:filt_opt.J-1)/filt_opt.Q);
    xi_psi = filt_opt['xi_psi']*np.power(2., (np.arange(0, -filt_opt['J'], step=-1, dtype=float)/filt_opt['Q']))
    sigma_psi = filt_opt['sigma_psi']*np.power(2., (np.arange(filt_opt['J'], dtype=float)/filt_opt['Q']))
    
    # Calculate linearly spaced band-pass filters so that they evenly
    # cover the remaining part of the spectrum
#     step = pi * 2^(-filt_opt.J/filt_opt.Q) * ...
#         (1-1/4*sigma0/filt_opt.sigma_phi*2^(1/filt_opt.Q))/filt_opt.P;
    if filt_opt['P'] > 0:
        step = np.pi * 2**(-float(filt_opt['J'])/filt_opt['Q']) * (1-1./4*sigma0/filt_opt['sigma_phi'] * 2**(1./filt_opt['Q']))/filt_opt['P']
    else:
        assert filt_opt['P'] >= 0
        step = 0.
#     xi_psi(filt_opt.J+1:filt_opt.J+filt_opt.P) = filt_opt.xi_psi * ...
#         2^((-filt_opt.J+1)/filt_opt.Q) - step * (1:filt_opt.P);
    xi_psi = np.concatenate((xi_psi, filt_opt['xi_psi']*np.power(2., float(-filt_opt['J']+1)/filt_opt['Q'])-step*np.r_[1:filt_opt['P']+1]))
#     sigma_psi(filt_opt.J+1:filt_opt.J+1+filt_opt.P) = ...
#         filt_opt.sigma_psi*2^((filt_opt.J-1)/filt_opt.Q);
    sigma_psi = np.concatenate((sigma_psi, np.ones(filt_opt['P']+1,dtype=float)*filt_opt['sigma_psi']*np.power(2., float(filt_opt['J']-1)/filt_opt['Q'])))
    #sigma_psi[(filt_opt['J']+1):(filt_opt['J']+filt_opt['P']+2)] = filt_opt['sigma_psi']*np.power(2.,[(filt_opt['J']-1)/filt_opt['Q']])

    #print sigma_psi[2]
    # Calculate band-pass filter
#     sigma_phi = filt_opt.sigma_phi * 2^((filt_opt.J-1)/filt_opt.Q);
# 
    sigma_phi = filt_opt['sigma_phi']*np.power(2., float(filt_opt['J']-1)/filt_opt['Q'])
    
    # Convert (spatial) sigmas to (frequential) bandwidths
#     bw_psi = pi/2 * sigma0./sigma_psi;
    bw_psi = np.pi/2 *sigma0/sigma_psi
    
#     if ~filt_opt.phi_dirac
#         bw_phi = pi/2 * sigma0./sigma_phi;
#     else
#         bw_phi = 2 * pi;
#     end
    if not filt_opt['phi_dirac']:
        bw_phi = np.pi/2 *sigma0/sigma_phi
    else:
        bw_phi = 2*np.pi
    bw_phi = np.atleast_1d(bw_phi)

    return xi_psi, bw_psi, bw_phi


def T_to_J(T, filt_opt):
    """
    Calculates the maximal wavelet scale J from T
    
    Usage
       J = T_to_J(T, filt_opt)
    
    Input
       T: A time interval T.
       filt_opt: A structure containing the parameters of a filter bank.
    
    Output
       J: The maximal wavelet scale J such that, for a filter bank created with
          the parameters in options and this J, the largest wavelet is of band-
          width approximately T.
    
    Description
       J is calculated using the formula T = 4B/phi_bw_multiplier*2^((J-1)/Q), 
       where B, phi_bw_multiplier and Q are taken from the filt_opt
       structure.
       Note that this function still works if T, filt_opt.Q and/or filt_opt.B
       are vectors. In this case, it provides a vector of the same length.
       This generalisation is useful when computing J for several layers in a
       scattering network, each layer having a different quality factor.
    
    See also
       DEFAULT_FILTER_OPTIONS, MORLET_FILTER_BANK_1D
    """

#    if  nargin ==2 && ~isstruct(filt_opt)
#        error('You must provide a filter options structure as second argument!');
#    end
    if not isinstance(filt_opt, dict):
        raise TypeError('You must provide a filter options structure as second argument!')
    
    filt_opt = dict(filt_opt)
#    
#     filt_opt = fill_struct(filt_opt,'Q',1);
#     filt_opt = fill_struct(filt_opt,'B',filt_opt.Q);
#     filt_opt = fill_struct(filt_opt,'phi_bw_multiplier', ...
#         1+(filt_opt.Q==1));

    filt_opt.setdefault('Q', np.asarray((1,)))
    filt_opt.setdefault('B', filt_opt['Q'])
    filt_opt.setdefault('phi_bw_multiplier', 1+(filt_opt['Q']==1))
# 
#     J = 1 + round(log2(T./(4*filt_opt.B./filt_opt.phi_bw_multiplier)) ...
#         .*filt_opt.Q);

    J = 1+np.asarray(np.round(np.log2(T/(4.*filt_opt['B']/filt_opt['phi_bw_multiplier']))*filt_opt['Q']), dtype=int)
    return J
# end


def gabor(N, xi, sigma, precision):
    extent = 1
    sigma = 1./sigma
    f = np.zeros((N,1), dtype=precision)
    seq = np.arange(N, dtype=precision).reshape(f.shape)
    for k in range(-extent,2+extent):
        f += np.exp(-np.square(((seq-k*N)/float(N))*2.*np.pi-xi)/(2.*sigma**2))
#        r = np.arange(0, N, dtype=precision)
#        r -= k*N
#        r /= float(N)
#        r *= 2*np.pi
#        r -= xi
#        np.square(r, out=r)
#        np.negative(r, out=r)
#        np.exp(r, out=r)
#        r /= 2*sigma**2
#        f += r
    return f
    
# function f = morletify(f,sigma)
def morletify(f, sigma):
#     f0 = f(1);
#     f = f-f0*gabor(length(f),0,sigma,class(f));
    f0 = f[0]
    return f-f0*gabor(len(f), 0, sigma, f.dtype)
# end


# function filter = optimize_filter(filter_f, lowpass, options)
def optimize_filter(filter_f, lowpass, options):
    """
     Optimize filter representation
    
     Usage
        filter = optimize_filter(filter_f, lowpass, options)
    
     Input
        filter_f (numeric): The Fourier transform of the filter.
        lowpass (boolean): If true, filter_f contains a lowpass filter.
        options (struct): Various options on how to optimize the filter:
           options.filter_format (char): Specifies the type of optimization, 
              either 'fourier', 'fourier_multires' or 'fourier_truncated'. See 
              description for more details.
           options.truncate_threshold (numeric): If options.filter_format is 
              'fourier_truncated', this indicates the threshold to be passed on 
              to TRUNCATE_FILTER. See the documentation of this function for more
              details.
    
     Output
        filter (struct or numeric): The optimized filter structure.
    
     Description
        Depending on the value of options.filter_format, OPTIMIZE_FILTER calls
        different functions to optimize the filter representation. If 'fourier',
        the function retains the Fourier representation of the filter. If 
        'fourier_multires', the filter is periodized and stored at all resolu-
        tions using PERIODIZE_FILTER. Finally, if it equals 'fourier_truncated',
        TRUNCATE_FILTER is called on filter_f.
    
     See also 
        PERIODIZE_FILTER, TRUNCATE_FILTER
    """

    options = dict(options)
#     options = fill_struct(options,'truncate_threshold',1e-3);
    options.setdefault('truncate_threshold', 1e-3)
#     options = fill_struct(options,'filter_format','fourier_multires');
    options.setdefault('filter_format', 'fourier_multires')
    
#     if strcmp(options.filter_format,'fourier')
#         filter = filter_f;
#     elseif strcmp(options.filter_format,'fourier_multires')
#         filter = periodize_filter(filter_f);
#     elseif strcmp(options.filter_format,'fourier_truncated')
#         filter = truncate_filter(filter_f,options.truncate_threshold,lowpass);
#     else
#         error(sprintf('Unknown filter format ''%s''',options.filter_format));
#     end
    if options['filter_format'] == 'fourier':
        filter_ = filter_f
#    elif options['filter_format'] == 'fourier_multires':
#        filter_ = periodize_filter(filter_f)
#    elif options['filter_format'] == 'fourier_truncated':
#        filter_ = truncate_filter(filter_f, options['truncate_threshold'], lowpass)
    else:
        raise NotImplementedError("Unknown filter format '%s'",options['filter_format'])
    return filter_
# end


def morlet_filter_bank_1d(sig_length, options=dict()):
    """
     Create a Morlet/Gabor filter bank
    
     Usage
        filters = morlet_filter_bank_1d(sz, options)
    
     Input
        sz (int): The size of the input data.
        options (struct, optional): Filter parameters, see below.
    
     Output
        filters (struct): The Morlet/Gabor filter bank corresponding to the data 
           size sz and the filter parameters in options.
    
     Description
        Depending on the value of options.filter_type, the functions either
        creates a Morlet filter bank (for filter_type 'morlet_1d') or a Gabor
        filter bank (for filter_type 'gabor_1d'). The former is obtained from 
        the latter by, for each filter, subtracting a constant times its enve-
        lopes such that the mean of the resulting function is zero.
    
        The following parameters can be specified in options:
           options.filter_type (char): See above (default 'morlet_1d').
           options.Q (int): The number of wavelets per octave (default 1).
           options.B (int): The reciprocal per-octave bandwidth of the wavelets 
              (default Q).
           options.J (int): The number of logarithmically spaced wavelets. For  
              Q=1, this corresponds to the total number of wavelets since there 
              are no  linearly spaced ones. Together with Q, this controls the  
              maximum extent the mother wavelet is dilated to obtain the rest of 
              the filter bank. Specifically, the largest filter has a bandwidth
              2^(J/Q) times that of the mother wavelet (default 
              T_to_J(sz, options)).
           options.phi_bw_multiplier (numeric): The ratio between the bandwidth 
              of the lowpass filter phi and the lowest-frequency wavelet (default
               2 if Q = 1, otherwise 1).
           options.boundary, options.precision, and options.filter_format: 
              See documentation for the FILTER_BANK function.
    
     See also
        SPLINE_FILTER_BANK_1D, FILTER_BANK
    """


#     if nargin < 2
#         options = struct();
#     end  
#     parameter_fields = {'filter_type','Q','B','J','P','xi_psi', ...
#         'sigma_psi', 'sigma_phi', 'boundary', 'phi_dirac'};
    parameter_fields = ('filter_type', 'Q', 'B', 'J', 'P', 'xi_psi',
                         'sigma_psi', 'sigma_phi', 'boundary', 'phi_dirac')
#     % If we are given a two-dimensional size, take first dimension
#     sig_length = sig_length(1);
    assert np.isscalar(sig_length) # it seems we are always scalar
#     sigma0 = 2/sqrt(3);
    sigma0 = 2/np.sqrt(3)
    
    options = dict(options) # copy
    
#     % Fill in default parameters
#     options = fill_struct(options, ...
#         'filter_type','morlet_1d');
    options.setdefault('filter_type', 'morlet_1d')
#     options = fill_struct(options, ...
#         'Q', 1);
    options.setdefault('Q', 1)
#     options = fill_struct(options, ...
#         'B', options.Q);
    options.setdefault('B', options['Q'])
#     options = fill_struct(options, ...
#         'xi_psi',1/2*(2^(-1/options.Q)+1)*pi);
    options.setdefault('xi_psi', 1/2.*(2**(-1./options['Q'])+1)*np.pi)  # 3.0112
#     options = fill_struct(options, ...
#         'sigma_psi',1/2*sigma0/(1-2^(-1/options.B)));
    options.setdefault('sigma_psi', 1/2.*sigma0/(1-2**(-1./options['B'])))
#     options = fill_struct(options, ...
#         'phi_bw_multiplier', 1+(options.Q==1));
    options.setdefault('phi_bw_multiplier', 1+(options['Q']==1))
#     options = fill_struct(options, ...
#         'sigma_phi', options.sigma_psi/options.phi_bw_multiplier);
    options.setdefault('sigma_phi' ,options['sigma_psi']/options['phi_bw_multiplier'])
#     options = fill_struct(options, ...
#         'J', T_to_J(sig_length, options));
    options.setdefault('J', T_to_J(sig_length, options))
#     options = fill_struct(options, ...
#         'P', round((2^(-1/options.Q)-1/4*sigma0/options.sigma_phi)/ ...  ...
#             (1-2^(-1/options.Q))));
    options.setdefault('P', np.round(
                                     (2**(-1./options['Q'])-1/4.*sigma0/options['sigma_phi'])/(1.-2**(-1./options['Q']))
                                     ).astype(int))
#     options = fill_struct(options, ...
#         'precision', 'double');
    options.setdefault('precision', 'double')
#     options = fill_struct(options, ...
#         'filter_format', 'fourier_truncated');
    options.setdefault('filter_format', 'fourier_truncated')
#     options = fill_struct(options, ...
#         'boundary', 'symm');
    options.setdefault('boundary', 'symm')
#     options = fill_struct(options, ...
#         'phi_dirac', 0);
    options.setdefault('phi_dirac', 0)
                    
#     if ~strcmp(options.filter_type,'morlet_1d') && ...
#             ~strcmp(options.filter_type,'gabor_1d')
#         error('Filter type must be ''morlet_1d'' or ''gabor_1d''.');
#     end
    if options['filter_type'] not in ('morlet_1d','gabor_1d'):
        raise ValueError("Filter type must be 'morlet_1d' or 'gabor_1d'")
            
#     do_gabor = strcmp(options.filter_type,'gabor_1d');
    do_gabor = (options['filter_type'] == 'gabor_1d')
#     filters = struct();
    filters = dict()
#     % Copy filter parameters into filter structure. This is needed by the
#     % scattering algorithm to calculate sampling, path space, etc.
#     filters.meta = struct();
    filters['meta'] = dict()
#     for l = 1:length(parameter_fields)
#         filters.meta = setfield(filters.meta,parameter_fields{l}, ...
#             getfield(options,parameter_fields{l}));
#     end
    for p in parameter_fields:
        filters['meta'][p] = options[p]
        
#     % The normalization factor for the wavelets, calculated using the filters
#     % at the finest resolution (N)
#     psi_ampl = 1;
    psi_ampl = 1
#     if (strcmp(options.boundary, 'symm'))
#         N = 2*sig_length;
#     else
#         N = sig_length;
#     end
    if options['boundary'] == 'symm':
        N = 2*sig_length
    else:
        N = sig_length
        
#     N = 2^ceil(log2(N));
    N = int(2**np.ceil(np.log2(N)))
#     filters.meta.size_filter = N;
    filters['meta']['size_filter'] = N    
#     filters.psi.filter = cell(1,options.J+options.P);
    filters['psi'] = dict(filter=[], meta=dict(k=[]))
#     filters.phi = [];
    filters['phi'] = dict()
#     [psi_center,psi_bw,phi_bw] = morlet_freq_1d(filters.meta);
    psi_center, psi_bw, phi_bw = morlet_freq_1d(filters['meta']) 
#     psi_sigma = sigma0*pi/2./psi_bw;
#     phi_sigma = sigma0*pi/2./phi_bw;
    psi_sigma = sigma0*np.pi/2./psi_bw
    phi_sigma = sigma0*np.pi/2./phi_bw 
#     % Calculate normalization of filters so that sum of squares does not
#     % exceed 2. This guarantees that the scattering transform is
#     % contractive.
#     S = zeros(N,1);
    S = np.zeros((N,1), dtype=float)
#     % As it occupies a larger portion of the spectrum, it is more
#     % important for the logarithmic portion of the filter bank to be
#     % properly normalized, so we only sum their contributions.
#     for j1 = 0:options.J-1
#         temp = gabor(N,psi_center(j1+1),psi_sigma(j1+1),options.precision);
#         if ~do_gabor
#             temp = morletify(temp,psi_sigma(j1+1));
#         end
#         S = S+abs(temp).^2;
#     end
    for j1 in range(options['J']):
        temp = gabor(N, psi_center[j1], psi_sigma[j1], options['precision'])
        if not do_gabor:
            temp = morletify(temp, psi_sigma[j1])
        np.abs(temp, out=temp)
        np.square(temp, out=temp)
        S += temp
#     psi_ampl = sqrt(2/max(S));
    psi_ampl = np.sqrt(2./np.max(S))

    # Apply the normalization factor to the filters.
#     for j1 = 0:length(filters.psi.filter)-1
#         temp = gabor(N,psi_center(j1+1),psi_sigma(j1+1),options.precision);
#         if ~do_gabor
#             temp = morletify(temp,psi_sigma(j1+1));
#         end
#         filters.psi.filter{j1+1} = optimize_filter(psi_ampl*temp,0,options);
#         filters.psi.meta.k(j1+1,1) = j1;
#     end
    krng = np.arange(options['J']+options['P']).reshape(-1,1)
    for j1 in krng[:,0]:
        temp = gabor(N, psi_center[j1], psi_sigma[j1], options['precision'])
        if not do_gabor:
            temp = morletify(temp, psi_sigma[j1])
        optflt = optimize_filter(psi_ampl*temp, 0, options)
        filters['psi']['filter'].append(optflt)
    filters['psi']['meta']['k'] = krng

    # Calculate the associated low-pass filter
#     if ~options.phi_dirac
#         filters.phi.filter = gabor(N, 0, phi_sigma, options.precision);
#     else
#         filters.phi.filter = ones(N,1,options.precision);
#     end     
    if not options['phi_dirac']:
        filters['phi']['filter'] = gabor(N, 0, phi_sigma, options['precision'])
    else:
        filters['phi']['filter'] = np.ones((N,1), dtype=options['precision'])

#     filters.phi.filter = optimize_filter(filters.phi.filter,1,options);
#     filters.phi.meta.k(1,1) = options.J+options.P;
#    filters['phi']['filter'] = optimize_filter(filters['phi']['filter'], 1, options)
    filters['phi']['meta'] = dict(k=options['J']+options['P'])
        
    return filters


def filters(Q, n, averaging=None, psi_min=None, psi_max=None, dtype=float):
    J = T_to_J(averaging or n, dict(Q=Q))
    options = dict(Q=Q, J=J, filter_format='fourier')

    flt = morlet_filter_bank_1d(n, options=options)

    phi = np.asarray(flt['phi']['filter'], dtype=dtype)[...,0]
    
    psi = np.asarray(flt['psi']['filter'], dtype=dtype)[...,0]
    psi_ixmin = 0 if psi_min is None else int(psi_min*n)
    psi_ixmax = n if psi_max is None else int(np.ceil(psi_max*n))
    argmaxs = [(pi,np.argmax(p)) for pi,p in enumerate(psi)] # from high ix to low ix
    ixhi = min(p for p,ix in argmaxs if ix <= psi_ixmax)
    ixlo = max(p for p,ix in argmaxs if ix >= psi_ixmin)
    psi = psi[ixhi:ixlo+1]
    return phi, psi


def filter_ds(kernel, threshold=1.e-3):
    if threshold:
        klen = len(kernel)
        kabs = np.abs(kernel[:klen//2+1])
        thr = np.max(kabs)*threshold
        above = np.where(kabs >= thr)[0]
        bw = max(np.max(above),np.max(above)-np.min(above))
        return 2**max(0, int(np.log2(klen/2./bw)))
    else:
        return 1


if __name__ == '__main__':
    phi, psi = filters(Q=8, n=8192, averaging=1024) # J=41,
    print("phi", phi.shape, phi)
    print("psi", psi.shape, psi)
