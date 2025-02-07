// xdp_lb.c
#include <arpa/inet.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <net/if.h>
#include "ecn.skel.h"  // The generated skeleton

int main() {
    //int map_fd = bpf_obj_get("sys/fs/bpf/mymap");
    //__u32 key=0;
    //long long initial_value =0;
    //bpf_map_update_elem(map_fd, &key, &initial_value, BPF_ANY);
    //int prog_fd;
    
    int ifindex = 2; 

    struct ecn_bpf *skel = ecn_bpf__open_and_load();
    

    bpf_program__attach_xdp(skel->progs.ecn, ifindex);

    // union bpf_attr attr = {};
    // attr.link_create.target_ifindex = ifindex;
    // attr.link_create.attach_type = BPF_XDP;
    // attr.link_create.prog_fd = bpf_get_prog_fd(skel->progs.ecn);
    // bpf_syscall(BPF_LINK_CREATE, attr);




    printf("Press Ctrl+C to exit...\n");
    while (1) {
        sleep(1);  // Keep the program running
    }

    // Cleanup and detach
    bpf_ecn_detach(ifindex, 0, NULL);
    xdp_ecn_bpf__detach(skel);
    xdp_ecn_bpf__destroy(skel);
    return 0;
}


