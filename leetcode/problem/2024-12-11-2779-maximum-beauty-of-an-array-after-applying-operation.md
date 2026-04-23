---
layout: leetcode-entry
title: "2779. Maximum Beauty of an Array After Applying Operation"
permalink: "/leetcode/problem/2024-12-11-2779-maximum-beauty-of-an-array-after-applying-operation/"
leetcode_ui: true
entry_slug: "2024-12-11-2779-maximum-beauty-of-an-array-after-applying-operation"
---

[2779. Maximum Beauty of an Array After Applying Operation](https://leetcode.com/problems/maximum-beauty-of-an-array-after-applying-operation/) medium
[blog post](https://leetcode.com/problems/maximum-beauty-of-an-array-after-applying-operation/solutions/6135417/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11122024-2779-maximum-beauty-of-an?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Va3W1jk4caQ)
[deep-dive](https://notebooklm.google.com/notebook/487335ba-82b3-4090-81ea-2246d47b9dc5/audio)
![1.webp](/assets/leetcode_daily_images/2c423e2d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/829

#### Problem TLDR

Max equal nums after adjusting to [-k..+k] #medium #binary_search #line_sweep

#### Intuition

Let's observe the data:

```j

    // 4 6 1 2      k=2
    // 2 4-1 0
    // 3 5 0 1
    // 4 6 1 2
    // 5 7 2 3
    // 6 8 3 4
    //[2..6] [6..8] [-1..3] [0..4]

    // -1 0 1 2 3 4 5 6 7 8
    //  s     * e
    //    s   * * e
    //        s *     e
    //                s   e
    //  1 2   3 3 2   2   1

    // -16   17   42   75   100
    //  [          ]
    //        [         ]
    //             [         ]
    //  s    s    e    e    e
    //            s

```

We can notice, each number is actually an interval of `[n-k..n+k]`. The task is to find maximum interval intersections.

This can be done in a several ways, one is to convert starts and ends, sort them, then do a line sweep with counter.

Another way is to search end index of `n + 2 * k`, we can do this with a binary search.

#### Approach

* we also can do a bucket sort for a line sweep, but careful with a zero point

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or O(n)

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun maximumBeauty(nums: IntArray, k: Int): Int {
        val se = mutableListOf<Pair<Int, Int>>()
        for (n in nums) { se += (n + k) to 1; se += (n - k) to -1 }
        se.sortWith(compareBy({ it.first }, { it.second }))
        var cnt = 0
        return se.maxOf { cnt -= it.second; cnt }
    }

```

```rust

    pub fn maximum_beauty(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort_unstable();
        (0..nums.len()).map(|i| {
            let (mut lo, mut hi) = (i + 1, nums.len() - 1);
            while lo <= hi {
                let m = (lo + hi) / 2;
                if nums[m] > nums[i] + k + k
                    { hi = m - 1 } else { lo = m + 1 }
            }; lo - i
        }).max().unwrap() as i32
    }

```

```c++

    int maximumBeauty(vector<int>& nums, int k) {
        int d[300002] { 0 }; int res = 1;
        for (int n: nums) ++d[n-k+100000], --d[n+k+100001];
        for (int i = 0, c = 0; i < 300002; ++i)
            res = max(res, c += d[i]);
        return res;
    }

```

