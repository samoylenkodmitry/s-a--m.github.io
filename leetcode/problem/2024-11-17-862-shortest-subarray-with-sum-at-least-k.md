---
layout: leetcode-entry
title: "862. Shortest Subarray with Sum at Least K"
permalink: "/leetcode/problem/2024-11-17-862-shortest-subarray-with-sum-at-least-k/"
leetcode_ui: true
entry_slug: "2024-11-17-862-shortest-subarray-with-sum-at-least-k"
---

[862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/) hard
[blog post](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/solutions/6054381/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17112024-862-shortest-subarray-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Yd9bBmurVqI)
[deep-dive](https://notebooklm.google.com/notebook/cee005a0-07e8-4ebb-9d5c-006d78cc995a/audio)
![1.webp](/assets/leetcode_daily_images/e386f217.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/803

#### Problem TLDR

Min subarray with sum at least `k` #hard #monotonic_queue #heap

#### Intuition

Side note:
Take me 1 hour and a hint about heap. Similar problem (https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/solutions/5355419/kotlin-rust/) was solved by me 5 months ago in 23 minutes (and then daily problems have the same two-pointers build up).

What should be noticed from examples:

```j

    // 0 1 2  3  4 5 6  7
    // 1 2 3 -3 -3 5 9  -3    k=14
    // 1 2 6  3  0 5 14 11
    //     *                  search for <= 6-14 <= -8
    //               *        search for <= 14-14 <= 0

```

We can use a cumulative sum to find a subarray sum. But as we search not strictly for the `k`, but for `at most k`, we should consider all keys less than `sum - k` and peek the most recent.

How to find the most recent? To do this we use another fact: we can safely remove all sums such `curr - sum >= k`, because no further addition to the `curr` will shrink already good interval.

Third trick is a monotonic queue instead of the heap to track the sums that are less than the current: keep queue increasing, with the `curr` on top.

#### Approach

* prefix sum can be in the same loop

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or O(n) for monotonic queue

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun shortestSubarray(nums: IntArray, k: Int): Int {
        var sum = 0L; var res = nums.size + 1
        val q = PriorityQueue<Pair<Long, Int>>(compareBy{ it.first })
        q.add(0L to -1)
        for ((i, n) in nums.withIndex()) {
            sum += n
            while (q.size > 0 && sum - q.peek().first >= k)
                res = min(res, i - q.poll().second)
            q += sum to i
        }
        return if (res > nums.size) -1 else res
    }

```
```rust

    pub fn shortest_subarray(nums: Vec<i32>, k: i32) -> i32 {
        let mut q = VecDeque::from([(0, -1)]);
        let (mut sum, mut res) = (0i64, i32::MAX);
        for (i, &n) in nums.iter().enumerate() {
            sum += n as i64;
            while q.front().is_some_and(|f| sum - f.0 >= k as i64)
                { res = res.min(i as i32 - q.pop_front().unwrap().1) }
            while q.back().is_some_and(|b| b.0 >= sum) { q.pop_back(); }
            q.push_back((sum, i as i32))
        }
        if res == i32::MAX { -1 } else { res }
    }

```
```c++

    int shortestSubarray(vector<int>& nums, int k) {
        long sum = 0; int res = nums.size() + 1;
        deque<pair<long, int>> q({ {0, -1} });
        for (int i = 0; i < nums.size(); q.push_back({sum, i++})) {
            sum += nums[i];
            while (!q.empty() && sum - q.front().first >= k)
                res = min(res, i - q.front().second), q.pop_front();
            while (!q.empty() && q.back().first >= sum) q.pop_back();
        }
        return res > nums.size() ? -1 : res;
    }

```

