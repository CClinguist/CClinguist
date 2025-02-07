// xdp_lb.bpf.c
#include <bpf/bpf_endian.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/tcp.h>
#include <linux/types.h>

#ifndef SOURCE_IP
#define SOURCE_IP 0x9DF0D123
#endif

static __always_inline __u16
csum_fold_helper(__u64 csum)
{
    int i;
    for (i = 0; i < 4; i++)
    {
        if (csum >> 16)
            csum = (csum & 0xffff) + (csum >> 16);
    }
    return ~csum;
}


static __always_inline __u16
iph_csum(struct iphdr *iph)
{
    iph->check = 0;
    unsigned long long csum = bpf_csum_diff(0, 0, (unsigned int *)iph, sizeof(struct iphdr), 0);
    return csum_fold_helper(csum);
}

//struct {
  //  __uint(type, BPF_MAP_TYPE_ARRAY);
   // __uint(max_entries, 1);
   // __type(key, __u32);         
   // __type(value, long long);       
//} my_map SEC(".maps");



int source_ip = bpf_htonl(SOURCE_IP);
SEC("xdp")
int ecn(struct xdp_md *ctx) {

    //bpf_printk("-----------1");
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    bpf_printk("xdp received packet");

    // Ethernet header
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    // Check if the packet is IP (IPv4)
    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    // IP header
    struct iphdr *iph = (struct iphdr *)(eth + 1);
    //bpf_printk("-----------2");
    if ((void *)(iph + 1) > data_end)
        return XDP_PASS;
    //modify the TOS field
    //bpf_printk("-----------3");
     bpf_printk("packet form IP 0x%x from IP 0x%x",
                bpf_ntohl(iph->daddr),
                bpf_ntohl(iph->saddr));
    if (iph->saddr != source_ip){
        return XDP_PASS;}
    bpf_printk("source matched");
    if ((iph->tos & 0x03)== 0x03){
	    bpf_printk("oldTos %u\n",iph->tos);
	    return XDP_PASS;}
    iph->tos = (iph->tos & ~0x03) | 0x02;
	// Update checksum
    iph->check = iph_csum(iph);
    bpf_printk("zsj-ecnMarking newTos %u\n",iph->tos);

  //  __u32 key = 0;
    
   // __u32 *value = bpf_map_lookup_elem(&my_map, &key);
   // if (value){
     //   __sync_fetch_and_add(value, 1);
       // bpf_printk("packets %u\n", *value);}
    
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";


