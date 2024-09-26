#define OWD 125
#define DROPPOINT 40
#define PRINTBUFF 0
#define true 1
#define false 0

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>
#include <linux/types.h>
#include <linux/netfilter.h>
#include <libnetfilter_queue/libnetfilter_queue.h>
#include <linux/tcp.h>
#include <linux/ip.h>
#include <sys/time.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <signal.h>
#include <time.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <linux/if_ether.h>
#include <netinet/ether.h>
#include <netinet/in.h>
#include <netinet/udp.h>

time_t current_time;
struct tm *time_info;
char time_string[20]; 

int drop = 1;
int acceptWindow = 0;
int dropWindow = 0;
unsigned int dropSeq = 0;
int cap = 50000;
int done = 0;
int emuDrop=10000;
uint32_t randomSeq=0;
int nextVal=0;
char buffSize[6];
int buff[2500];
int indx = 0;

int maxseq=0;

int dropped[2500];

_Bool isRetrans(int seq) {
    int i = 0;
    for (i = 0; i < dropWindow; i++) {
        if (seq == dropped[i])
            return 1;
    }
    return 0;
}

char* itoa(int n, char* number) {
    if (number == NULL) {
        return NULL;
    }

    int i = 0;
    int j;
    int temp = n;
    int digit;

    if (n == 0) {
        number[i++] = '0';
        number[i] = '\0';
        return number;
    }

    int isNegative = 0;
    if (n < 0) {
        isNegative = 1;
        temp = -n;
    }

    while (temp != 0) {
        digit = temp % 10;
        number[i++] = (char) (digit + '0');
        temp /= 10;
    }

    if (isNegative) {
        number[i++] = '-';
    }

    number[i] = '\0';
    for (j = 0; j < i / 2; j++) {
        char temp_char = number[j];
        number[j] = number[i - 1 - j];
        number[i - 1 - j] = temp_char;
    }

    return number;
}

void print_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm *time_info = localtime(&tv.tv_sec);

    int minutes = time_info->tm_min;
    int seconds = time_info->tm_sec;

    long microseconds = tv.tv_usec;

    printf("Current time: %02d:%02d.%06ld\n", minutes, seconds, microseconds);
}

void split(char string[], int start, int end) {
    char str[10];
    int i = 0;
    for (i = 0; i < (end - start); i++) {
        str[i] = string[start + i];
    }
    strcpy(buffSize, str);
    return;
}

int getBuff() {
    char filename[] = "/proc/net/netfilter/nfnetlink_queue";
    FILE *file = fopen(filename, "r");

    fseek(file, 0, SEEK_SET);
    if (file != NULL) {
        char stats[60];
        fgets(stats, sizeof stats, file);
        split(stats, 12, 18);
    }
    return atoi(buffSize);
}

void print_raw_data(const unsigned char *data, int length) {
    for (int i = 0; i < length; i++) {
        printf("%02x ", data[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    if (length % 16 != 0) {
        printf("\n");
    }
}

void print_ip_header(struct iphdr *iph) {
    struct in_addr src_addr, dst_addr;
    src_addr.s_addr = iph->saddr;
    dst_addr.s_addr = iph->daddr;

    printf("    Raw Source IP: 0x%08x, Converted: %s\n", ntohl(iph->saddr), inet_ntoa(src_addr));
    printf("    Raw Destination IP: 0x%08x, Converted: %s\n", ntohl(iph->daddr), inet_ntoa(dst_addr));
    printf("    Raw Protocol: 0x%02x, Converted: %d\n", iph->protocol, iph->protocol);
}

void print_tcp_header(struct tcphdr *tcph) {
    printf("    Raw Sequence Number: 0x%08x, Converted: %u\n", ntohl(tcph->seq), ntohl(tcph->seq));
    printf("    Raw Acknowledgment Number: 0x%08x, Converted: %u\n", ntohl(tcph->ack_seq), ntohl(tcph->ack_seq));
}

void parse_packet(unsigned char *pkt, int pkt_len) {
    struct iphdr *iph = (struct iphdr *)pkt;
    print_ip_header(iph);

    int ip_header_len = iph->ihl * 4;
    unsigned char *transport_header = pkt + ip_header_len;

    if (iph->protocol == IPPROTO_TCP) {
        struct tcphdr *tcph = (struct tcphdr *)transport_header;
        print_tcp_header(tcph);
    } else {
        printf("Unsupported protocol: %d\n", iph->protocol);
    }
}

typedef struct {
    unsigned int seq_num;
    unsigned int ip_len;
    unsigned int payload_len;
} threeValues;

threeValues get_seqNum_len(unsigned char *pkt, int pkt_len) {
    threeValues result;
    result.seq_num = 0;
    result.payload_len = 0;
    result.ip_len = 0;
    struct iphdr *iph = (struct iphdr *)pkt;
    int total_len = ntohs(iph->tot_len);
    result.ip_len = total_len;
    int ip_header_len = iph->ihl * 4;
    unsigned char *transport_header = pkt + ip_header_len;

    if (iph->protocol == IPPROTO_TCP) {
        struct tcphdr *tcph = (struct tcphdr *)transport_header;
        int tcp_header_len = tcph->doff * 4;
        result.payload_len = total_len - (ip_header_len + tcp_header_len);
        result.seq_num = ntohl(tcph->seq);
    } else if (iph->protocol == IPPROTO_UDP) {
        struct udphdr *udph = (struct udphdr *)transport_header;
        int udp_header_len = sizeof(struct udphdr);
        int udp_total_length = ntohs(udph->len);
        result.payload_len = udp_total_length;
        unsigned char *quic_data = (unsigned char *)udph + udp_header_len;

        if (result.payload_len > 0) {
            result.seq_num = (quic_data[0] << 24) | (quic_data[1] << 16) | (quic_data[2] << 8) | quic_data[3];
        }
    }
    return result;
}

void print_timeval(struct timeval tv) {
    char buffer[32];
    struct tm *tm_info = gmtime(&tv.tv_sec);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    printf("Time: %s.%06ld UTC\n", buffer, tv.tv_usec);
}

void get_current_utc_time(char *buffer, size_t buffer_size) {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm *utc_time = gmtime(&tv.tv_sec);
    strftime(buffer, buffer_size, "%H:%M:%S", utc_time);

    size_t len = strlen(buffer);
    snprintf(buffer + len, buffer_size - len, ".%06ld", tv.tv_usec);
}

void get_relative_time(char *buffer, size_t buffer_size) {
    static struct timeval start_time;
    struct timeval current_time;
    gettimeofday(&current_time, NULL);

    if (start_time.tv_sec == 0 && start_time.tv_usec == 0) {
        start_time = current_time;
    }

    long seconds = current_time.tv_sec - start_time.tv_sec;
    long microseconds = current_time.tv_usec - start_time.tv_usec;

    if (microseconds < 0) {
        microseconds += 1000000;
        seconds -= 1;
    }

    snprintf(buffer, buffer_size, "%ld.%06ld", seconds, microseconds);
}

void destroySession(struct nfq_handle *h, struct nfq_q_handle *qh) {
    nfq_destroy_queue(qh);
    #ifdef INSANE
        nfq_unbind_pf(h, AF_INET);
    #endif

    nfq_close(h);
}

static int cb(struct nfq_q_handle *qh, struct nfgenmsg *nfmsg,
              struct nfq_data *nfa, void *data) {
    unsigned char *pkt;
    struct nfqnl_msg_packet_hdr *header;
    uint32_t id = 0;
    uint32_t tseq = 0;
    unsigned int seq_num = 0;

    header = nfq_get_msg_packet_hdr(nfa);
    id = ntohl(header->packet_id);
    unsigned int ret = nfq_get_payload(nfa, &pkt);

    if (ret >= 0) {
        char time_buffer[80];
        get_current_utc_time(time_buffer, sizeof(time_buffer));

        threeValues result;
        result = get_seqNum_len(pkt, ret);
        printf("%s %u %u %u\n", time_buffer, result.ip_len, result.payload_len, result.seq_num);
    }

    int i = 24;
    for (i = 24; i < 28; i++) {
        tseq |= (unsigned int)pkt[i] << (8 * (27 - i));
    }

    if (tseq == randomSeq) {
        return nfq_set_verdict(qh, id, NF_ACCEPT, 0, NULL);
    } else {
        if (acceptWindow == DROPPOINT) {
            acceptWindow++;
            randomSeq = tseq;
            return nfq_set_verdict(qh, id, NF_DROP, 0, NULL);
        } else {
            acceptWindow++;
            return nfq_set_verdict(qh, id, NF_ACCEPT, 0, NULL);
        }
    }
}

int getWinSize(char line[]) {
    char num[4];
    int i = 0, j = 0;
    for (i = 0; i < 6; i++) {
        if (line[i] == ' ')
            break;
    }
    for (j = 0; j < 4; j++) {
        num[j] = line[i + j + 1];
    }
    return atoi(num);
}

void extract_domain(const char *url, char *domain) {
    domain[0] = '\0';

    if (strncmp(url, "http://", 7) == 0) {
        sscanf(url, "http://%255[^/]", domain);
    } else if (strncmp(url, "https://", 8) == 0) {
        sscanf(url, "https://%255[^/]", domain);
    } else {
        sscanf(url, "%255[^/]", domain);
    }
}

int main(int argc, char **argv) {
    struct nfq_handle *h;
    struct nfq_q_handle *qh;
    int fd;
    int rv;
    char buf[4096] __attribute__ ((aligned));
    int lastWindow;
    int inputting = 0;

    h = nfq_open();
    if (!h) {
        exit(1);
    }

    if (nfq_unbind_pf(h, AF_INET) < 0) {
        fprintf(stderr, "error during nfq_unbind_pf()\n");
        exit(1);
    }

    if (nfq_bind_pf(h, AF_INET) < 0) {
        fprintf(stderr, "error during nfq_bind_pf()\n");
        exit(1);
    }

    qh = nfq_create_queue(h, 0, &cb, NULL);
    if (!qh) {
        fprintf(stderr, "error during nfq_create_queue()\n");
        exit(1);
    }

    if (nfq_set_mode(qh, NFQNL_COPY_PACKET, 0xffff) < 0) {
        fprintf(stderr, "can't set packet_copy mode\n");
        exit(1);
    }

    fd = nfq_fd(h);
    int counter = 0;

    int delay = atoi(argv[4]);
    int nextDelay = atoi(argv[5]);
    int switchPoint = atoi(argv[6]);

    signal(SIGCHLD, SIG_IGN);
    pid_t pid = fork();

    if (pid == 0) {
        char get[8192];
        char *url = argv[1];
        char *domain = argv[2];
        char *ip_address = argv[3];

        int port = 443;
        if (strncmp(url, "http://", 7) == 0) {
            port = 80;
        }

        int ret = snprintf(get, sizeof(get), 
            "curl -L -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:62.0) Gecko/20100101 Firefox/62.0' "
            "--connect-timeout 10 --max-time 30 "
            "-o indexPage --header 'Host: %s' --resolve %s:%d:%s \"%s\"",
            domain, domain, port, ip_address, url);

        if (ret < 0 || ret >= sizeof(get)) {
            fprintf(stderr, "Error: curl command buffer overflow or encoding error.\n");
            return 1;
        }

        system(get);
        
        exit(0);
    } else {
        int status = -1;
        while (done == 0 && (rv = recv(fd, buf, sizeof(buf), 0)) && rv >= 0) {
            usleep(delay);
            nfq_handle_packet(h, buf, rv);
            if (counter > switchPoint) delay = nextDelay;
            counter++;

            status = kill(pid, 0);
            if (status == 0) {
                continue;
            } else {
                done = 1;
                break;
            }
        }

        destroySession(h, qh);

        if (done == -1)
            exit(0);
    }    

    return 0; 
}
