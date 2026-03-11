import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_pickle('out-shakespeare-char-passthrough/timing_data.pkl')
df_base = pd.read_pickle('out-shakespeare-char-timetest/timing_data.pkl')

df_gpu = pd.read_pickle('out-shakespeare-char-passthrough-gpu/timing_data.pkl')
df_gpu_base = pd.read_pickle('out-shakespeare-char-gpu/timing_data.pkl')

# CPU plotting

# iter vs time
plt.plot(df['iter_num'], df['time'], label='Passthrough')
plt.plot(df_base['iter_num'], df_base['time'], label='Base')
plt.legend()

plt.savefig('timings.png')

plt.close()

# iter vs train loss
plt.plot(df['iter_num'], df['train_loss'], label='Passthrough')
plt.plot(df_base['iter_num'], df_base['train_loss'], label='Base')
plt.legend()

plt.savefig('train_loss.png')

plt.close()

# iter vs val loss
plt.plot(df['iter_num'], df['val_loss'], label='Passthrough')
plt.plot(df_base['iter_num'], df_base['val_loss'], label='Base')
plt.legend()

plt.savefig('val_loss.png')

plt.close()

# time vs val loss
plt.plot(df['time'], df['val_loss'], label='Passthrough')
plt.plot(df_base['time'], df_base['val_loss'], label='Base')
plt.legend()

plt.savefig('val_loss_time.png')

plt.close()

# time vs train loss
plt.plot(df['time'], df['train_loss'], label='Passthrough')
plt.plot(df_base['time'], df_base['train_loss'], label='Base')
plt.legend()

plt.savefig('train_loss_time.png')

plt.close()

# GPU plotting

# iter vs time (GPU)
plt.plot(df_gpu['iter_num'], df_gpu['time'], label='Passthrough GPU')
plt.plot(df_gpu_base['iter_num'], df_gpu_base['time'], label='Base GPU')
plt.legend()

plt.savefig('time_gpu.png')

plt.close()

# iter vs val loss (GPU)
plt.plot(df_gpu['iter_num'], df_gpu['val_loss'], label='Passthrough GPU')
plt.plot(df_gpu_base['iter_num'], df_gpu_base['val_loss'], label='Base GPU')
plt.legend()

plt.savefig('val_loss_gpu.png')

plt.close()

# iter vs train loss (GPU)
plt.plot(df_gpu['iter_num'], df_gpu['train_loss'], label='Passthrough GPU')
plt.plot(df_gpu_base['iter_num'], df_gpu_base['train_loss'], label='Base GPU')
plt.legend()

plt.savefig('train_loss_gpu.png')

plt.close()

# time vs val loss (GPU)
plt.plot(df_gpu['time'], df_gpu['val_loss'], label='Passthrough GPU')
plt.plot(df_gpu_base['time'], df_gpu_base['val_loss'], label='Base GPU')
plt.legend()

plt.savefig('val_loss_time_gpu.png')

plt.close()

# time vs train loss (GPU)
plt.plot(df_gpu['time'], df_gpu['train_loss'], label='Passthrough GPU')
plt.plot(df_gpu_base['time'], df_gpu_base['train_loss'], label='Base GPU')
plt.legend()

plt.savefig('train_loss_time_gpu.png')

plt.close()