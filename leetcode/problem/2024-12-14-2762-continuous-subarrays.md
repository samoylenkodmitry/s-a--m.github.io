---
layout: leetcode-entry
title: "2762. Continuous Subarrays"
permalink: "/leetcode/problem/2024-12-14-2762-continuous-subarrays/"
leetcode_ui: true
entry_slug: "2024-12-14-2762-continuous-subarrays"
---

[2762. Continuous Subarrays](https://leetcode.com/problems/continuous-subarrays/description/) medium
[blog post](https://leetcode.com/problems/continuous-subarrays/solutions/6145675/kotlin-rust-by-samoylenkodmitry-qp6q/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14122024-2762-continuous-subarrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ksBDeTXMi7g)
[deep-dive](https://notebooklm.google.com/notebook/f1cbf33e-71e0-426d-8cb2-8d7719a457b1/audio)
![1.webp](/assets/leetcode_daily_images/e5e2da85.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/832

#### Problem TLDR

Count subarrays with difference <= 2 #medium #monotonic_queue #tree_set

#### Intuition

Observe the example, let's use two pointers sliding window:

```j

    // 5 4 2 4  min max   iq    dq
    // i         5   5    5     5
    // j
    //   i       4   5    4     5 4
    //     i     2   5    2     5 4 2 shrink
    //   j       2   4    2     4 2   new max

```
After we shrink the window by moving `j`, we should update `min` and `max` of the window. To keep all potential next maximums and minimums we can use a monotonic queue technique: remove all non-increasing/non-decreasing values.

Another approach is to use a TreeSet: it naturally would give us updated `min` and `max`.

#### Approach

* if we drop the duplicates, the max queue size would be 4
* to use Kotlin's TreeSet, we should also preserve duplicates by storing the indices

#### Complexity

- Time complexity:
$$O(n)$$ or O(nlog(n))

- Space complexity:
$$O(1)$$ or O(n)

#### Code

```kotlin

    fun continuousSubarrays(nums: IntArray): Long {
        val s = TreeSet<Pair<Int, Int>>(compareBy({it.first}, {it.second}))
        var j = 0
        return nums.withIndex().sumOf { (i, n) ->
            s += n to i
            while (s.last().first - s.first().first > 2) s -= nums[j] to j++
            1L + i - j
        }
    }

```

```rust

    pub fn continuous_subarrays(nums: Vec<i32>) -> i64 {
        let (mut iq, mut dq, mut j) = (VecDeque::new(), VecDeque::new(), 0);
        nums.iter().enumerate().map(|(i, &n)| {
            while iq.back().map_or(false, |&b| nums[b] >= n) { iq.pop_back(); }
            while dq.back().map_or(false, |&b| nums[b] <= n) { dq.pop_back(); }
            iq.push_back(i); dq.push_back(i);
            while n - nums[*iq.front().unwrap()] > 2 { j = iq.pop_front().unwrap() + 1 }
            while nums[*dq.front().unwrap()] - n > 2 { j = dq.pop_front().unwrap() + 1 }
            1 + i as i64 - j as i64
        }).sum()
    }

```
```c++

    long long continuousSubarrays(vector<int>& n) {
        long long r = 0; multiset<int> s;
        for (int i = 0, j = 0; i < n.size(); ++i) {
            s.insert(n[i]);
            while (s.size() && *s.rbegin() - *s.begin() > 2)
                s.erase(s.find(n[j++]));
            r += i - j + 1;
        }; return r;
    }

```

