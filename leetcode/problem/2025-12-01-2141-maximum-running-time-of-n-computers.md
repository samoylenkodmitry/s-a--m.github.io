---
layout: leetcode-entry
title: "2141. Maximum Running Time of N Computers"
permalink: "/leetcode/problem/2025-12-01-2141-maximum-running-time-of-n-computers/"
leetcode_ui: true
entry_slug: "2025-12-01-2141-maximum-running-time-of-n-computers"
---

[2141. Maximum Running Time of N Computers](https://leetcode.com/problems/maximum-running-time-of-n-computers/description) hard
[blog post](https://leetcode.com/problems/maximum-running-time-of-n-computers/solutions/7385022/kotlin-rust-by-samoylenkodmitry-4lyi/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01122025-2141-maximum-running-time?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VKBlvh1hzHI)

![98398bb6-add2-4ff1-92a0-4c90aa42af56 (1).webp](/assets/leetcode_daily_images/deb0a0ac.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1190

#### Problem TLDR

Max time to run n computers with batteries #hard #bs

#### Intuition

```j
    // n=3     b=[3,3,3] t= 3
    // n=3     b=[3,3,3,3] t= 3
    //            2 2 2 3
    //              1 1 2
    //            1   0 1
    //            0 0   0 t=4
    // b= 1 2 3 4 5 6   n=3
    //    0 1 2
    //      0 1 3
    //        0 2 4
    //          1 3 5
    //          0 2 4 non optimal, left 6 which is 2*n
    //
    //          3 4 5
    //          2 3 4
    //          1 2 3
    //          0 1 2
    //    0 1 2
    //      0 1   0     1+2 left, non optimal, 1+2=n
    //
    //    0       4 5
    //      1     3 4
    //      0   3   3
    //        2 2 2
    //        1   1 2
    //          1 0 1
    //        0 0   0   7 optimal, left is 0
    //                  so the algo: take all bigger and one small
    //                  does this always work?
    //
    // n=2  1 40 40    just take biggest
    // ok looks like algo is not obvious, is it binary search?
    //
    // we have time, answer question can run?
    //               true true true | false false false
    //
    // now, given the time how to consume it?
    //
    // 1 2 3 4 5 6  n=3  time=7 sum=21;
    //                   any minute we burn 3 points
    //                           total = 7*3 = 21
    //                   time=8 total < sum
    // 3 3 3 n=2 time=5; sum=9    total=2*5=10
    //           time=4           total=2*4=8
    //
    // corner cases
    // 1 1 10 n=2 sum=12  maxtime=6
    // 2 2 10 n=2 sum=14
    //            let t = 4, 4*2 = 8
    // 2+2+4 = 8
    //
    //  1 40 40 n=2, let t=40, 2*40=80
    //
    // 30minute, look for hint: already knew;
    // the actual hardness is how to determine if can run

```

With given time answer `canRun` n computers.
Total required energy is `t*n`.
Each battery can't give more than it have or more than `t` (it can't be alone used in parallel).

#### Approach

* another intuition is: average `sum/n`, take max while it is bigger than average (they are not relevant to detection of max time); the leftover `sum/n` is the answer

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 13ms
    fun maxRunTime(n: Int, b: IntArray): Long {
        var lo = 0L; var hi = Long.MAX_VALUE/n
        while (lo <= hi) {
            val t = lo + (hi - lo) / 2
            if (t*n <= b.sumOf { min(1L*it, t) }) lo = t + 1 else hi = t - 1
        }
        return hi
    }
```
```rust
// 14ms
    pub fn max_run_time(n: i32, b: Vec<i32>) -> i64 {
        let (mut lo, mut hi) = (0, i64::MAX / n as i64);
        while lo <= hi {
            let t = lo + (hi - lo) / 2;
            if t * n as i64 <= b.iter().map(|&x|t.min(x as i64)).sum::<i64>()
            { lo = t + 1 } else { hi = t - 1 }
        }; hi
    }
```

