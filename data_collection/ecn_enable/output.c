/*i
 * This code is used for OPER packets replacement at linux-based switches (tested on kernel 3.13, 
 * no warranty, modification may be required, on other version of kernal),
 * cooperate with 'iptables' configuration or NETFILTER modules.
 * @author Jesson LIU
 * any reuse of this code follows the GPL protocol
 */
#include <linux/if_ether.h>
#include <stdio.h>
#include <assert.h>
#include <netinet/in.h>
#include <linux/types.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/netfilter.h>
#include <libnetfilter_queue/libnetfilter_queue.h>
#include <pthread.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <linux/ip.h>
#include <string.h>
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
//#include <netinet/inet_ecn.h>  //zsj add

#define QUEUE_NUMBER_	80
#define QUEUE_LEN	50
#define QUEUE_THR	20
#define TRUE		1
#define FALSE		0
#define DATA_PKT	1
#define OPER_PKT	0
#define TCP_PRO		0x06
#define UDP_PRO		0x11
#define OPER_PRO	0xFD

/* Parameters of netfilter queues */
struct	nfq_handle *h_;
struct	nfq_q_handle *qh_;
struct	nfnl_handle *nh_;
int	fd_;
int	rv_;
char	buf_[4096];

u_int8_t isReplaceable_;

//static bool  force;
# define __force	__attribute__((force))
typedef unsigned short u16;
typedef unsigned int u32;

static pthread_mutex_t lock_;

struct PacketInfo{
	u_int32_t id;		// Packet id in netfilter queue
	u_int8_t  type;		// 1: data packets; 2: OPER packets
	struct PacketInfo *previous;
	struct PacketInfo *next;
};

u_int32_t size_;
u_int32_t packetNum_;
struct PacketInfo *head_, *firstOPER_, *tail_;



//=======================LOGIC OPERATION FINISHED============================
 void* enqueuePKT(){

	printf("enqueue listener thread started...\n");
	
	// receive
	for(;;){
		if ((rv_ = recv(fd_, buf_, sizeof(buf_), 0)) >= 0){
			nfq_handle_packet(h_, buf_, rv_);
			continue;
		}
	}
	/*while ((rv_ = recv(fd_, buf_, sizeof(buf_), 0)) && rv_ >= 0) {
		nfq_handle_packet(h_, buf_, rv_);
	}*/
}

static int cb(struct nfq_q_handle *qh, struct nfgenmsg *nfmsg,
        struct nfq_data *nfa, void *data)
{
	u_int32_t id = 0;
	u_int32_t enif, deif;
	u_int8_t  isPKTadd = FALSE;
	printf("---------------ecn start ------------------------\n\n");
	struct nfqnl_msg_packet_hdr *ph;
	printf("---------------here ------------------------\n\n");
	printf("nfa: %p\n", nfa);
	ph = nfq_get_msg_packet_hdr(nfa);	
	
	if (ph) {
		id = ntohl(ph->packet_id);
	}
	
	// To distinguish the forwarding interface
	enif = nfq_get_indev(nfa);//zsj nfa where it is defined
	deif = nfq_get_outdev(nfa);
	
	/* Get packet information */
	struct iphdr *iph;
	struct tcphdr *tcph;
	struct udphdr *udph;
	int ret;
	char *nf_packet;
	printf("---------------here 1------------------------\n\n");
	
	ret = nfq_get_payload(nfa, (unsigned char**)&nf_packet);
	iph = (struct iphdr *)nf_packet;

	//u32 oldTos=iph->tos;
        printf("---------------here 2------------------------\n\n");
	

	//if(htons(iph->tot_len)>500){
	//	packetNum_++;
	//}

	
	//ret = nfq_get_payload(nfa, (unsigned char**)&nf_packet);
	
	
	//iph = ((struct iphdr *)nf_packet);
	
	printf("zsj-ecnMarking newTos %u\n",iph->tos);

    nfq_set_verdict(qh_, id, NF_ACCEPT, ret, nf_packet);
	return 1;
}

int main()
{
	printf("------------------zsj-main----------------------\n");
	printf("     Linux-based OPER switch v1.0 \n");
	printf(" @author Jesson LIU (Jesson.liu@qq.com)\n");
	printf("  copy and rewrite under GPL protocol\n");
	printf("----------------------------------------\n");
	printf("\nstarting initialization of netfilter queue...\n");
	

	// initialization of netfilter queue
	h_=nfq_open();
	if(!h_){
		printf("error during nfq_open\n");
		return 0;
	}
	
	if(nfq_unbind_pf(h_, AF_INET)<0)
	{
		printf("error during nfq_unbind_pf\n");
		return 0;
	}
	
	if(nfq_bind_pf(h_, AF_INET)<0)
	{
		printf("error during nfq_bind_pf\n");
		return 0;
	}
	
	qh_ = nfq_create_queue(h_, QUEUE_NUMBER_, &cb, NULL);
	if(!qh_){
		printf("error during nfq_create_queue\n");
		return 0;
	}
	
	if(nfq_set_mode(qh_, NFQNL_COPY_PACKET, 0xffff) <0){
		printf("error during nfq_set_mode\n");
		return 0;
	}
	fd_ = nfq_fd(h_);

	// initialization of lock
	size_ = 0;
	head_ = NULL;
	tail_ = NULL;

	printf("netfilter queue initialization complete!\n\n");


	pthread_mutex_init(&lock_, NULL);

	//pthread_t forwarding_thread;
	//pthread_create(&forwarding_thread, NULL, forwardPackets, NULL);	

	pthread_t enqueue_thread;
	pthread_create(&enqueue_thread, NULL, enqueuePKT, NULL);
		

	//pthread_join(forwarding_thread, NULL);
	pthread_join(enqueue_thread, NULL);
	pthread_mutex_destroy(&lock_);

	nfq_destroy_queue(qh_);
	
	nfq_close(h_);
	return 0;
}

