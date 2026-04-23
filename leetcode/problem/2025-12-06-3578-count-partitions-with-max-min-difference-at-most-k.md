---
layout: leetcode-entry
title: "3578. Count Partitions With Max-Min Difference at Most K"
permalink: "/leetcode/problem/2025-12-06-3578-count-partitions-with-max-min-difference-at-most-k/"
leetcode_ui: true
entry_slug: "2025-12-06-3578-count-partitions-with-max-min-difference-at-most-k"
---

[3578. Count Partitions With Max-Min Difference at Most K](https://leetcode.com/problems/count-partitions-with-max-min-difference-at-most-k/description/) medium
[blog post](https://leetcode.com/problems/count-partitions-with-max-min-difference-at-most-k/solutions/7395692/kotlin-rust-by-samoylenkodmitry-5qwx/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06122025-3578-count-partitions-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1BlCD2lICno)

![bc2b490c-f8ed-4c2c-b10b-7605dcde8233 (1).webp](/assets/leetcode_daily_images/5ccc0bcf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1195

#### Problem TLDR

Ways to partition into parts (max-min) at most k #medium #dp #monotonic_queue

#### Intuition

Didn't solve.

```j
   // at most means 0..k
    // so for each position i
    // look for all possible positions j
    // abs(n[j]-n[i]) at most k
    // or it is every position j..i
    // and move j if n[j] is not in n[i]-k..n[i]+k
    //
    // corner case 3 3 4    k=0/1
    // completely wrong solution
    //
    // 4 1 3 7
    // j i
    // j     i    when i goes to 7 it is out of bounds with 1
    //            but j stays at 4 and is ok
    //
    // from '1' look at ..0 and 5.. left and right
    // let's only look left
    // 9 4 1 3 7  k=4     TreeMap lookup? have to check all ranges 5+,0-
    // i          1
    //   * i      2 1*2=2
    //   * * i    3 1*3
    //       * i  2 2*3
    //
    // 24 minute lets' look for hints, ok its dp
    // f(9 4 1 3) = f(9)*f(4 1 3)
    // 1 2 3 4 5 6    k=2
    // * * *
    //   * * *
    //     * * *
    //       * * *   24=6x4     * * * = 6 = 3*4/2
    // 32 minute hint 2&3, so the second trick is running min/max
    //                                            this is a hard problemm
    // 1 hour mark
    // my algo correclty detects j
    // but i don't know how to count ways, tests are not passing
    // look for solutions
    // from lee solution, its not just about detecting j position
    //                    we should track accumulated values
    //                    and subtract them while we moving j

```

* i is the end of the window
* j is the left position, such that max - min at most k
* dp[i] = dp[j] + dp[j+1] + dp[j+2] + ... + dp[i-1]
* use prefix sum of dp to calculate sum(dp[j..i])
* use decreasing queue to find max
* use increasing queue to find min
* pop both queues together while moving j

#### Approach

* this problem is hard

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 105ms
    fun countPartitions(n: IntArray, k: Int): Int {
        val dp = IntArray(n.size+1); dp[0] = 1; val M = 1000000007
        val ps = IntArray(n.size+1); ps[0] = 1; var j = 0
        val qmax = ArrayDeque<Int>(); val qmin = ArrayDeque<Int>()
        for (i in n.indices) {
            while (qmax.size > 0 && n[qmax.last()] < n[i]) qmax.removeLast()
            while (qmin.size > 0 && n[qmin.last()] > n[i]) qmin.removeLast()
            qmax += i; qmin += i
            while (n[qmax.first()]-n[qmin.first()] > k) {
                if (qmax.first() < ++j) qmax.removeFirst()
                if (qmin.first() < j) qmin.removeFirst()
            }
            dp[i+1] = (M + ps[i] - if (j > 0) ps[j-1] else 0) % M
            ps[i+1] = (ps[i] + dp[i + 1]) % M
        }
        return dp[n.size]
    }
```

