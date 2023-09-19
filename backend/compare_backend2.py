import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
import pickle

def listdir_nohidden(path):
	for f in os.listdir(path):
		if not f.startswith('.'):
			yield f

def get_fits(path1, channel, extra=None, include=True):
	""" Get all fits that

	"""
	if channel == 'ch1':
		snip = '_ch1'
	else:
		snip = '_ch2'
	MODES = []
	PATHS = []
	PARAMS = []
	TMP = []
	lst = listdir_nohidden(path1)

	i=0
	for fname in lst:
		fits = [fname for fname in listdir_nohidden(path1+fname) if snip in fname]
		print(fits)
		for fit in fits:
			if extra == None:
				TMP.extend([fit])
				resultpath = [fname for fname in listdir_nohidden(path1+fname+'/'+fit) if np.logical_and('Bestfit' in fname, '.pkl' in fname)]
				paramspath = [fname for fname in listdir_nohidden(path1+fname+'/'+fit) if 'Params.npy' in fname]
			elif (extra!=None and include==True):
				if extra in fit:
					TMP.extend([fit])
					resultpath = [fname for fname in listdir_nohidden(path1+fname+'/'+fit) if np.logical_and('Bestfit' in fname, '.pkl' in fname)]
					paramspath = [fname for fname in listdir_nohidden(path1+fname+'/'+fit) if 'Params.npy' in fname]
				else:
					resultpath = []
			else:
				if extra not in fit:
					TMP.extend([fit])
					resultpath = [fname for fname in listdir_nohidden(path1+fname+'/'+fit) if np.logical_and('Bestfit' in fname, '.pkl' in fname)]
					paramspath = [fname for fname in listdir_nohidden(path1+fname+'/'+fit) if 'Params.npy' in fname]
				else:
					resultpath = []
			if resultpath != []:
				# path to RESULTS.npy file
				pathres = path1+fname+'/'+fit+'/'+resultpath[0]
				pathpar = path1+fname+'/'+fit+'/'+paramspath[0]
				MODES.append(fit)
				PATHS.append(pathres)
				PARAMS.append(pathpar)
	return MODES, PATHS, TMP, PARAMS

def checklist(mega_keys, cols, rows, MODES, TMP, fname='checklist1_ch1.pdf', supkey=[]):
	""" First table checklist for channel 1 with ecc and w fixed.
	"""

	# make subplots
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, len(rows)))
	fig.suptitle('Checklist of SPCA Fits with ecc and w fixed', y=1.01, fontsize=14, color='firebrick')
	
	# annotate each columns and rows
	pad = 5 # in points
	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')

	# print fits for which we have a pkl file
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			if supkey!=[]:
				keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			#print(cols[col], rows[row], 'keys:', keys)
			#print('anti_keys', anti_keys)
			for mode in MODES:
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keys
				if np.bool(flag)==True:
					axes[row, col].text(0, 0.5, mode[:8]+'\n'+mode[8:26]+'\n'+mode[26:])
					axes[row, col].set_xticks([])
					axes[row, col].set_yticks([])

	# print fits with existing fits but not pickle
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			if supkey!=[]:
				keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			for mode in TMP:
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keysqstat
				if np.bool(flag)==True:
					axes[row, col].text(0, 0.1, mode[:8], color ='C0')

	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/home/ldang05/projects/def-ncowan/ldang05/Spitzer_Data/KELT-20b/compare_fits/'+fname
	fig.savefig(fpath, bbox_inches='tight')
	return

def get_param_values(mega_keys, cols, rows, MODES, PARAMS, par, supkey = None):
	PAR = []
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			if supkey != None:
				keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			for i in range(len(MODES)):
				mode = MODES[i]
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
					# rewrite keys
				if np.bool(flag)==True:
					params = np.load(PARAMS[i])
					PAR.append(params[par][0])
	parmin, parmax = np.array(PAR).min(), np.array(PAR).max()
	return parmin, parmax

def lightcurve_grid(mega_keys, cols, rows, MODES, TMP, PATHS, PARAMS, evimin=0, evimax=0, chi2min=0, 
	chi2max=0, par='', weight='chi2B',supkey=[], mintime = 56417.2414096537, maxtime = 56420.8356839537, 
	minflux = 0.9995, maxflux = 1.0029, deci=2, channel='ch1', printx = 2, printx2 = 1.5):
	
	# picking colors
	if channel=='ch1':
		R1, G1, B1, dark = 56/255, 166/255, 237/255, 'darkblue'
		R2, G2, B2, dark = 43/255, 177/255, 127/255, 'darkblue'

	else:
		R1, G1, B1, dark = 255/255, 79/255, 99/255, 'darkred'
		R2, G2, B2, dark = 255/255, 99/255, 71/255, 'darkred'

	# starting plot
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13*len(rows)/8))
	fig.suptitle(weight+' of SPCA Fits for '+channel, y=1.01, fontsize=14, color=dark)

	pad = 5 # in points

	##### labeling colums and rows
	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			if supkey!=[]:
				keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			for i in range(len(MODES)):
				mode = MODES[i]
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keys
				if np.bool(flag)==True:
					params = np.load(PARAMS[i])
					# printing and changing the background color to reflect the goodness of fit
					if weight=='chi2B':
						delta  = params['chi2B'][0] - chi2min
						chi2 = str(round(delta,2))
						chi2dif = chi2max-chi2min
						alpha  = round((chi2max-params['chi2B'][0])/chi2dif, 4)
						axes[row,col].set_facecolor((R1, G1, B1, alpha))
						axes[row, col].text(mintime+printx, 1.0021, chi2, color=dark)
					elif weight=='evidenceB':
						delta  = params['evidenceB'][0] - evimax
						evidence = str(round(delta,2))
						evidif = evimax-evimin
						alpha  = round((params['evidenceB'][0]-evimin)/evidif, 4)
						axes[row,col].set_facecolor((R2, G2, B2, alpha))
						axes[row, col].text(mintime+printx, 1.0021, evidence, color=dark)
					# printing the value of params you care about
					if par=='ecc':
						if 'fix' not in mode:
							ecosw    = params['ecosw'][0]
							esinw    = params['esinw'][0]
							value    = np.sqrt(ecosw**2 + esinw**2) 
							valuep   = par[:3]+':'+str(round(value,deci))
						else:
							value    = np.sqrt(0.27005**2 + (-0.0612)**2)
							valuep   = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0025, valuep, color='black')
					elif par=='w':
						if 'fix' not in mode:
							ecosw    = params['ecosw'][0]
							esinw    = params['esinw'][0]
							value    = np.arctan2(esinw, ecosw)*180./np.pi
							valuep   = par[:3]+':'+str(round(value,deci))
						else:
							value    = np.arctan2(-0.0612, 0.27005)*180./np.pi
							valuep   = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0025, valuep, color='black')
					elif par!='':
						value    = params[par][0]
						valuep = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0025, valuep, color='black')
					# plotting the lightcurve
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color=dark)
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row,col].set_xticks([])

	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/Evidence_Grids/lightcurve_grid_'+weight+'_'+str(supkey)[1:-1]+'_'+channel+'_'+par+'.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return


def lightcurve_grid_peri(mega_keys, cols, rows, MODES, TMP, PATHS, PARAMS, evimin=0, evimax=0, chi2min=0, 
	chi2max=0, par='', weight='chi2B',supkey=[], mintime = 56417.2414096537, maxtime = 56420.8356839537, 
	minflux = 0.9995, maxflux = 1.0029, deci=2, channel='ch1', printx = 2, printx2 = 1.5):
	
	# picking colors
	if channel=='ch1':
		R1, G1, B1, dark = 220/255, 50/255, 67/255, '#850a16'#56/255, 166/255, 237/255, 'darkblue'
		R2, G2, B2, dark = 220/255, 50/255, 67/255, '#850a16'#43/255, 177/255, 127/255, 'darkblue'

	else:
		R1, G1, B1, dark = 249/255, 161/255, 33/255, 'chocolate'#255/255, 79/255, 99/255, 'darkred'
		R2, G2, B2, dark = 249/255, 161/255, 33/255, 'chocolate'#255/255, 99/255, 71/255, 'darkred'

	# starting plot
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13*len(rows)/8))
	#fig.suptitle(weight+' of SPCA Fits for '+channel, y=1.01, fontsize=14, color=dark)

	pad = 5 # in points

	##### labeling colums and rows
	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	fancyrows = ['1st order', '2nd order', '1st order \n w/ prior on '+r'$f_p$', '2nd order \n w/ prior on '+r'$f_p$']
	for ax, row in zip(axes[:,0], fancyrows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center', rotation=90)

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			if supkey!=[]:
				keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			for i in range(len(MODES)):
				mode = MODES[i]
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keys
				if np.bool(flag)==True:
					params = np.load(PARAMS[i])
					# printing and changing the background color to reflect the goodness of fit
					if weight=='chi2B':
						delta  =  params['chi2B'][0] - chi2min
						chi2 = str(np.abs(round(delta,2)))
						chi2dif = chi2max-chi2min
						alpha  = round((chi2max-params['chi2B'][0])/chi2dif, 4)
						axes[row,col].set_facecolor((R1, G1, B1, alpha))
						axes[row, col].text(mintime+printx, 1.0025, chi2, color='k')
					elif weight=='evidenceB':
						delta  =  params['evidenceB'][0] - evimax
						evidence = str(np.abs(round(delta,2)))
						evip     = r'$\Delta E$ = ' + evidence
						evidif = evimax-evimin
						alpha  = round((params['evidenceB'][0]-evimin)/evidif, 4)
						axes[row,col].set_facecolor((R2, G2, B2, alpha))
						axes[row, col].text(mintime+printx, 1.0025, evip, color='k')
					elif weight=='bicB':
						delta  =  -2*(params['evidenceB'][0] - evimax)
						evidence = str(np.abs(round(delta,2)))
						evip     = r'$\Delta BIC$ = ' + evidence
						evidif = evimax-evimin
						alpha  = round((params['evidenceB'][0]-evimin)/evidif, 4)
						axes[row,col].set_facecolor((R2, G2, B2, alpha))
						axes[row, col].text(mintime+printx, 1.0025, evip, color='k')
					# printing the value of params you care about
					if par=='ecc':
						if 'fix' not in mode:
							ecosw    = params['ecosw'][0]
							esinw    = params['esinw'][0]
							value    = np.sqrt(ecosw**2 + esinw**2) 
							valuep   = par[:3]+'='+str(round(value,deci))
						else:
							value    = np.sqrt(0.27005**2 + (-0.0612)**2)
							valuep   = par[:3]+'='+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color=dark)
					elif par=='w':
						if 'fix' not in mode:
							ecosw    = params['ecosw'][0]
							esinw    = params['esinw'][0]
							value    = np.arctan2(esinw, ecosw)*180./np.pi
							valuep   = par[:3]+'='+str(round(value,deci))
						else:
							value    = np.arctan2(-0.0612, 0.27005)*180./np.pi
							valuep   = par[:3]+'='+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color=dark)
					elif par=='fp':
						value    = params[par][0]*1000
						valuep = par[:3]+'='+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color=dark)
					elif par!='':
						value    = params[par][0]
						valuep = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color=dark)
					# plotting the lightcurve
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color=dark)
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row,col].set_xticks([])
					axes[row,col].axvline(x=params['t0'][0]+2.504744446305267-params['per'][0], color='darkgrey', lw=0.7)
					#if np.isin(row, np.arange(0,4)):
					#	axes[row,col].axvline(x=params['t0'][0]+2.6549013199983165-params['per'][0], color='lightgrey', lw=0.7)
					#elif np.isin(row, np.arange(4,8)):
					#	axes[row,col].axvline(x=params['t0'][0]+2.6037840864009922-params['per'][0], color='lightgrey', lw=0.7)
					#elif np.isin(row, np.arange(8,12)):
					#	axes[row,col].axvline(x=params['t0'][0]+2.504744446305267-params['per'][0], color='lightgrey', lw=0.7)

	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/Evidence_Grids/lightcurve_grid_'+weight+'_'+str(supkey)[1:-1]+'_'+channel+'_'+par+'.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return


def lightcurve_grid_veto(mega_keys, cols, rows, MODES, TMP, PATHS, PARAMS, evimin=0, evimax=0, chi2min=0, 
	chi2max=0, par='', weight='chi2B',supkey=['veto'], mintime = 56417.2414096537, maxtime = 56420.8356839537, 
	minflux = 0.9995, maxflux = 1.0029, deci=2, channel='ch1', printx = 2, printx2 = 1.5):
	
	# picking colors
	if channel=='ch1':
		R1, G1, B1, dark = 220/255, 50/255, 67/255, '#850a16'#56/255, 166/255, 237/255, 'darkblue'
		R2, G2, B2, dark = 220/255, 50/255, 67/255, '#850a16'#43/255, 177/255, 127/255, 'darkblue'

	else:
		R1, G1, B1, dark = 249/255, 161/255, 33/255, 'chocolate'#255/255, 79/255, 99/255, 'darkred'
		R2, G2, B2, dark = 249/255, 161/255, 33/255, 'chocolate'#255/255, 99/255, 71/255, 'darkred'

	# starting plot
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13*len(rows)/8))
	#fig.suptitle(weight+' of SPCA Fits for '+channel, y=1.01, fontsize=14, color=dark)

	pad = 5 # in points

	##### labeling colums and rows
	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')
	fancyrows = ['1st order', '2nd order', '1st order \n w/ prior on '+r'$f_p$', '2nd order \n w/ prior on '+r'$f_p$']
	for ax, row in zip(axes[:,0], fancyrows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center', rotation=90)

	# for col in range(len(cols)):
	# 	keyscol = cols[col].split('_')
	# 	for row in range(len(rows)):
	# 		keys = keyscol.copy()
	# 		keys.extend(rows[row].split('_'))
	# 		if supkey!=[]:
	# 			keys.extend(supkey)
	# 		#keys.extend(['veto'])                
	# 		anti_keys = [key for key in mega_keys if key not in keys]
	# 		for i in range(len(MODES)):
	# 			mode = MODES[i]
	# 			flag = True
	# 			for key in anti_keys:
	# 				flag *= (key not in mode)
	# 			for key in keys:
	# 				flag *= (key in mode)
	# 			# rewrite keys
	# 			if np.bool(flag)==True:
	# 				params = np.load(PARAMS[i])
	# 				# plotting the lightcurve
	# 				pkl = pickle.load(open(PATHS[i], 'rb'))
	# 				axes[row,col].plot(pkl[1], pkl[3], color='grey', lw=0.7)                      
                        
        
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			if supkey!=[]:
				keys.extend(supkey)
			keys.extend(['veto'])
			anti_keys = [key for key in mega_keys if key not in keys]
			for i in range(len(MODES)):
				mode = MODES[i]
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keys
				if np.bool(flag)==True:
					params = np.load(PARAMS[i])
					# printing and changing the background color to reflect the goodness of fit
					if weight=='chi2B':
						delta  = params['chi2B'][0] - chi2min
						chi2 = str(round(delta,2))
						chi2dif = chi2max-chi2min
						alpha  = round((chi2max-params['chi2B'][0])/chi2dif, 4)
						axes[row,col].set_facecolor((R1, G1, B1, alpha))
						axes[row, col].text(mintime+printx, 1.0025, chi2, color=dark)
					elif weight=='evidenceB':
						delta  =  params['evidenceB'][0] - evimax
						evidence = str(np.abs(round(delta,2)))
						evip     = r'$\Delta E$ = ' + evidence
						evidif = evimax-evimin
						alpha  = round((params['evidenceB'][0]-evimin)/evidif, 4)
						axes[row,col].set_facecolor((R2, G2, B2, alpha))
						axes[row, col].text(mintime+printx, 1.0025, evip, color='k')
					elif weight=='bicB':
						delta  =  -2*(params['evidenceB'][0] - evimax)
						evidence = str(np.abs(round(delta,2)))
						evip     = r'$\Delta BIC$ = ' + evidence
						evidif = evimax-evimin
						alpha  = round((params['evidenceB'][0]-evimin)/evidif, 4)
						axes[row,col].set_facecolor((R2, G2, B2, alpha))
						axes[row, col].text(mintime+printx, 1.0025, evip, color='k')
					# printing the value of params you care about
					if par=='ecc':
						if 'fix' not in mode:
							ecosw    = params['ecosw'][0]
							esinw    = params['esinw'][0]
							value    = np.sqrt(ecosw**2 + esinw**2) 
							valuep   = par[:3]+':'+str(round(value,deci))
						else:
							value    = np.sqrt(0.27005**2 + (-0.0612)**2)
							valuep   = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color='black')
					elif par=='w':
						if 'fix' not in mode:
							ecosw    = params['ecosw'][0]
							esinw    = params['esinw'][0]
							value    = np.arctan2(esinw, ecosw)*180./np.pi
							valuep   = par[:3]+':'+str(round(value,deci))
						else:
							value    = np.arctan2(-0.0612, 0.27005)*180./np.pi
							valuep   = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color='black')
					elif par=='fp':
						value    = params[par][0]*1000
						valuep = par[:3]+'='+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color=dark)
					elif par!='':
						value    = params[par][0]
						valuep = par[:3]+':'+str(round(value,deci))
						axes[row,col].text(mintime+printx2, 1.0021, valuep, color='black')
					# plotting the lightcurve
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color=dark)
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row,col].set_xticks([])
					axes[row,col].axvline(x=params['t0'][0]+2.504744446305267-8*params['per'][0], color='darkgrey', lw=0.7)
					#if np.isin(row, np.arange(0,4)):
					#	axes[row,col].axvline(x=params['t0'][0]+2.6549013199983165-8*params['per'][0], color='white', lw=0.7)
					#elif np.isin(row, np.arange(4,8)):
					#	axes[row,col].axvline(x=params['t0'][0]+2.6037840864009922-8*params['per'][0], color='white', lw=0.7)
					#elif np.isin(row, np.arange(8,12)):
					#	axes[row,col].axvline(x=params['t0'][0]+2.504744446305267-8*params['per'][0], color='white', lw=0.7)

	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/Evidence_Grids/lightcurve_grid_'+weight+'_'+str(supkey)[1:-1]+'_'+channel+'_'+par+'.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return