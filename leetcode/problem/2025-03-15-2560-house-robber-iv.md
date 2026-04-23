---
layout: leetcode-entry
title: "2560. House Robber IV"
permalink: "/leetcode/problem/2025-03-15-2560-house-robber-iv/"
leetcode_ui: true
entry_slug: "2025-03-15-2560-house-robber-iv"
---

[2560. House Robber IV](https://leetcode.com/problems/house-robber-iv/description/) medium
[blog post](https://leetcode.com/problems/house-robber-iv/solutions/6538164/kotlin-rust-by-samoylenkodmitry-5m3o/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15032025-2560-house-robber-iv?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/sFg-DGlQZi8)
![1.webp](/assets/leetcode_daily_images/0fa47f99.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/928

#### Problem TLDR

Min-max of non-adjucents in array #medium #binary_search

#### Intuition

This is a binary search week, but how to apply it to this problem?

One way of thinking is to search in a space of the `capability`: we should have k elements, all no bigger the choosen capability.

```j

    // 2 3 5 9     m capability = max(n[i])
    //               should have k elements, all <= m

```

This is a half of the problem. Now the trickiest part, the robbing (let's assume we are not robbing, but `giving out the money`, how about that?).

Forturnately, the brain dead greedy solution just works: always choose the current if possible, don't think about the future, you will handle it when the time comes.

#### Approach

* try the greedy, it is simpler to just check if it works, than to spend the time on a DP and get the TLE
* several ways to write the greedy part: boolean flag `ban`, `can`, or adjusting the iterator pointer

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minCapability(nums: IntArray, k: Int): Int {
        var lo = 0; var hi = Int.MAX_VALUE
        while (lo <= hi) {
            val m = lo + (hi - lo) / 2; var b = false
            val c = nums.count { b = !b && it <= m; b }
            if (c >= k) hi = m - 1 else lo = m + 1
        }
        return lo
    }

```
```rust

    pub fn min_capability(nums: Vec<i32>, k: i32) -> i32 {
        let (mut lo, mut hi) = (0, i32::MAX);
        while lo <= hi {
            let m = lo + (hi - lo) / 2;
            let (mut cnt, mut can) = (0, true);
            for &x in &nums { if (can && x <= m) { cnt += 1; can = false }
                else { can = true }}
            if cnt >= k { hi = m - 1 } else { lo = m + 1 }
        } lo
    }

```
```c++

    int minCapability(vector<int>& nums, int k) {
        int lo = 0, hi = INT_MAX;
        while (lo <= hi) {
            int m = lo + (hi - lo) / 2, cnt = 0;
            for (int i = 0; i < size(nums); ++i)
                if (nums[i] <= m) cnt++, i++;
            cnt < k ? lo = m + 1 : hi = m - 1;
        } return lo;
    }

```

