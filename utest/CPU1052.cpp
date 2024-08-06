#include <vector>

class CPU1052 {
public:
    int maxSatisfied(std::vector<int>& customers, std::vector<int>& grumpy, int minutes) {
        int ans1 = 0;
        int ans2 = 0;
        for (int i=0;i<grumpy.size();++i) {
            if (grumpy[i] == 0) ans1 += customers[i];
        }

        int local = 0;
        for (int i=0;i<grumpy.size();++i) {
            local += (grumpy[i]==1)?customers[i]:0;
            if (i>=minutes) {
                local -= (grumpy[i-minutes]==1)?customers[i-minutes]:0;
            }
            ans2 = std::max(ans2, local);
        }
        return ans1+ans2;
    }
};