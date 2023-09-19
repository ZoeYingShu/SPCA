import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time
import pickle

def get_fits_ch1(path1):
	""" Get all fits that

	"""
	MODES = []
	PATHS = []
	PARAMS = []
	TMP = []
	lst = os.listdir(path1)

	i=0
	for fname in lst:
		fits = [fname for fname in os.listdir(path1+fname) if '_ch1' in fname]
		for fit in fits:
			if 'TSLOPE' not in fit:
				TMP.extend([fit])
				resultpath = [fname for fname in os.listdir(path1+fname+'/'+fit) if np.logical_and('Bestfit' in fname, '.pkl' in fname)]
				paramspath = [fname for fname in os.listdir(path1+fname+'/'+fit) if 'Params.npy' in fname]
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

def get_fits_ch2(path1):
	""" Get all fits that

	"""
	MODES = []
	PATHS = []
	PARAMS = []
	TMP = []
	lst = os.listdir(path1)

	i=0
	for fname in lst:
		fits = [fname for fname in os.listdir(path1+fname) if '_ch2' in fname]
		for fit in fits:
			if 'HSIDE' in fit:
				TMP.extend([fit])
				resultpath = [fname for fname in os.listdir(path1+fname+'/'+fit) if np.logical_and('Bestfit' in fname, '.pkl' in fname)]
				paramspath = [fname for fname in os.listdir(path1+fname+'/'+fit) if 'Params.npy' in fname]
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

def checklist_1(mega_keys, MODES, TMP):
	""" First table checklist for channel 1 with ecc and w fixed.
	"""
	# keys to get all fits with ecc and w fixed
	supkey = ['ecc_w','fix']

	# rows and columns element
	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']
	rows = ['v1', 'v2', 'v1_veto', 'v2_veto', 'v1_fp', 'v2_fp', 
			'v1_veto_fp', 'v2_veto_fp']

	# make subplots
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 8))
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
			keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
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

	# print fits with existing fits beut not pickle
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
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

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/checklist1_ch1.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def checklist_2(mega_keys, MODES, TMP):
	""" First table checklist for channel 1 with ecc and w with prior.
	"""
	# keys to get all fits with ecc and w fixed
	supkey = ['ecc_w']

	# rows and columns element
	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']
	rows = ['v1', 'v2', 'v1_veto', 'v2_veto', 'v1_fp', 'v2_fp', 
			'v1_veto_fp', 'v2_veto_fp']

	# make subplots
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 8))
	fig.suptitle('Checklist of SPCA Fits with prior on ecc and w', y=1.01, fontsize=14, color='firebrick')
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
			keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
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

	# print fits with existing fits beut not pickle
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
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

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/checklist2_ch1.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def checklist_3(mega_keys, MODES, TMP):
	""" First table checklist for channel 1 with no ecc and w prior.
	"""

	# rows and columns element
	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']
	rows = ['v1', 'v2', 'v1_veto', 'v2_veto', 'v1_fp', 'v2_fp', 
			'v1_veto_fp', 'v2_veto_fp']

	# make subplots
	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 8))
	fig.suptitle('Checklist of SPCA Fits with no prior on ecc and w', y=1.01, fontsize=14, color='firebrick')
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
			anti_keys = [key for key in mega_keys if key not in keys]
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

	# print fits with existing fits beut not pickle
	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
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

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/checklist3_ch1.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def checklist_ch2(mega_keys, MODES, TMP):

	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']
	rows = ['v1', 'v2', 'v1_fp', 'v2_fp', 'v1_ecc', 'v2_ecc', 'v1_fp_ecc', 'v2_fp_ecc', 
			'v1_ecc_fix', 'v2_ecc_fix', 'v1_fp_ecc_fix', 'v2_fp_ecc_fix']


	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13))


	pad = 5 # in points

	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')
		
	# printing mode (is SPCA run exist)
	#MODES = [mode for mode in MODES if 'ecc_w_fix' in mode]

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			#keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			for mode in MODES:
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keysqstat
				if np.bool(flag)==True:
					axes[row, col].text(0, 0.5, mode[:13]+'\n'+mode[13:26]+'\n'+mode[26:])
					axes[row, col].set_xticks([])
					axes[row, col].set_yticks([])

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
			#keys.extend(supkey)
			anti_keys = [key for key in mega_keys if key not in keys]
			for mode in TMP:
				flag = True
				for key in anti_keys:
					flag *= (key not in mode)
				for key in keys:
					flag *= (key in mode)
				# rewrite keysqstat
				if np.bool(flag)==True:
					axes[row, col].text(0, 0.1, mode[:13], color ='C0')



	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/checklist_ch2.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def get_evidence(mega_keys, cols, rows, MODES, PARAMS, supkey = None):
	EVI = []
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
					EVI.append(params['evidenceB'][0])
	evimin, evimax = np.array(EVI).min(), np.array(EVI).max()
	return evimin, evimax

def get_chi2(mega_keys, cols, rows, MODES, PARAMS, supkey = None):
	EVI = []
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
					EVI.append(params['chi2B'][0])
	evimin, evimax = np.array(EVI).min(), np.array(EVI).max()
	return evimin, evimax

def get_fp(mega_keys, cols, rows, MODES, PARAMS, supkey = None):
	EVI = []
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
					EVI.append(params['fp'][0])
	evimin, evimax = np.array(EVI).min(), np.array(EVI).max()
	return evimin, evimax

def lightcurve_grid_ch2(mega_keys, cols, rows, MODES, TMP, PATHS, PARAMS, evimin, evimax, par='chi2B', supkey=[]):
	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']

	rows = ['v1', 'v2', 'v1_fp', 'v2_fp', 'v1_ecc', 'v2_ecc', 'v1_fp_ecc', 'v2_fp_ecc', 
			'v1_ecc_fix', 'v2_ecc_fix', 'v1_fp_ecc_fix', 'v2_fp_ecc_fix']


	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13))
	fig.suptitle(par+' of SPCA Fits for ch2', y=1.01, fontsize=14, color='firebrick')

	pad = 5 # in points

	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')
		
	mintime = 56417.2414096537
	maxtime = 56420.8356839537

	minflux = 0.9995
	maxflux = 1.0029

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
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
					if par=='chi2B':
						delta  = params['chi2B'][0] - evimin
						evidence = str(round(delta,2))
						evidif = evimax-evimin
						alpha  = round((evimax-params['chi2B'][0])/evidif, 4)
						axes[row,col].set_facecolor((255/255, 79/255, 99/255, alpha))
					elif par=='evidenceB':
						delta  = params[par][0] - evimax
						evidence = str(round(delta,2))
						evidif = evimax-evimin
						alpha  = round((params[par][0]-evimin)/evidif, 4)
						#print(evimin, evimax, delta)
						axes[row,col].set_facecolor((255/255, 99/255, 71/255, alpha))
					elif par=='fp':
						evimin1 =1000*evimin
						evimax1 =1000*evimax
						delta  = 1000*params[par][0]
						evidence = str(round(delta,2))
						evidif = evimax1-evimin1
						alpha  = round((1000*params[par][0]-evimin1)/evidif, 4)
						axes[row,col].set_facecolor((252/255, 172/255, 13/255, alpha))
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color='darkred', label = evidence)
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row,col].set_xticks([])
					axes[row, col].text(mintime+0.1, 1.0021, evidence, color='darkred')
	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/lightcurve_grid'+str(supkey)+'_ch2_'+par+'.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def lightcurve_grid1_ch1(mega_keys, cols, rows, MODES, TMP, PATHS, PARAMS, evimin, evimax, par='evidenceB', supkey=[]):

	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13))
	fig.suptitle(par+' of SPCA Fits with ecc and w fixed', y=1.01, fontsize=14, color='royalblue')

	pad = 5 # in points

	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')
		
	mintime = 56394.90810715371
	maxtime = 56398.4838244537

	minflux = 0.9995
	maxflux = 1.003

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
					if par=='chi2B':
						delta  = params['chi2B'][0] - evimin
						evidence = str(round(delta,2))
						evidif = evimax-evimin
						alpha  = round((evimax-params['chi2B'][0])/evidif, 4)
						axes[row,col].set_facecolor((56/255, 166/255, 237/255, alpha))
					elif par=='evidenceB':
						delta  = params[par][0] - evimax
						evidence = str(round(delta,2))
						evidif = evimax-evimin
						alpha  = round((params[par][0]-evimin)/evidif, 4)
						axes[row,col].set_facecolor((56/255, 166/255, 237/255, alpha))
					elif par=='fp':
						evimin1 =1000*evimin
						evimax1 =1000*evimax
						delta  = 1000*params[par][0]
						evidence = str(round(delta,2))
						evidif = evimax1-evimin1
						alpha  = round((1000*params[par][0]-evimin1)/evidif, 4)
						axes[row,col].set_facecolor((4/255, 144/255, 158/255, alpha))
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color='darkblue', label = evidence)
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row,col].set_xticks([])
					axes[row, col].text(mintime+2, 1.0, evidence, color='darkblue')
	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/lightcurve_grid'+str(supkey)+'_ch1_'+par+'.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def lightcurve_grid2_ch1(mega_keys, MODES, TMP, PATHS):
	supkey = ['ecc_w']

	# rows and columns element
	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']
	rows = ['v1', 'v2', 'v1_veto', 'v2_veto', 'v1_fp', 'v2_fp', 
			'v1_veto_fp', 'v2_veto_fp']


	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13))
	fig.suptitle('Checklist of SPCA Fits with priors on ecc and w', y=1.01, fontsize=14, color='firebrick')

	pad = 5 # in points

	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')
		
	mintime = 56394.90810715371
	maxtime = 56398.4838244537

	minflux = 0.9995
	maxflux = 1.003

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
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
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color='coral')
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row, col].set_xticks([])

	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/lightcurve_grid2_ch1.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return

def lightcurve_grid3_ch1(mega_keys, MODES, TMP, PATHS):

	# rows and columns element
	cols = ['Poly2', 'Poly3', 'Poly4', 'Poly5', 'BLISS', 
			'PLDAper1_3x3', 'PLDAper2_3x3', 'PLDAper1_5x5', 'PLDAper2_5x5']
	rows = ['v1', 'v2', 'v1_veto', 'v2_veto', 'v1_fp', 'v2_fp', 
			'v1_veto_fp', 'v2_veto_fp']


	fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), sharex=True, sharey=True, figsize=(15, 13))
	fig.suptitle('Checklist of SPCA Fits with no priors on ecc and w', y=1.01, fontsize=14, color='firebrick')

	pad = 5 # in points

	for ax, col in zip(axes[0], cols):
		ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
					xycoords='axes fraction', textcoords='offset points',
					size='large', ha='center', va='baseline')

	for ax, row in zip(axes[:,0], rows):
		ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')
		
	mintime = 56394.90810715371
	maxtime = 56398.4838244537

	minflux = 0.9995
	maxflux = 1.003

	for col in range(len(cols)):
		keyscol = cols[col].split('_')
		for row in range(len(rows)):
			keys = keyscol.copy()
			keys.extend(rows[row].split('_'))
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
					pkl = pickle.load(open(PATHS[i], 'rb'))
					axes[row,col].plot(pkl[1], pkl[3], color='coral')
					axes[row,col].set_xlim(mintime, maxtime)
					axes[row,col].set_ylim(minflux, maxflux)
					axes[row, col].set_xticks([])

	fig.tight_layout()
	fig.subplots_adjust(left=0.15, top=0.95, hspace=0, wspace=0)

	fpath = '/Users/ldang/Desktop/XO-3b-EBM/lightcurve_grid3_ch1.pdf'
	fig.savefig(fpath, bbox_inches='tight')
	return