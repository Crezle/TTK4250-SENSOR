if __name__ == '__main__':
    
    WHAT = 'sim' # TODO: 'sim' or 'real'
    
    if WHAT == 'real':
        from run_real_SLAM import main
    elif WHAT == 'sim':
        from run_simulated_SLAM import main
    else:
        raise ValueError(f"what = {WHAT} should be 'real' or 'sim'")
    main()