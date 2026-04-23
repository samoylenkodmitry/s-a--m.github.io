---
layout: leetcode-entry
title: "2071. Maximum Number of Tasks You Can Assign"
permalink: "/leetcode/problem/2025-05-01-2071-maximum-number-of-tasks-you-can-assign/"
leetcode_ui: true
entry_slug: "2025-05-01-2071-maximum-number-of-tasks-you-can-assign"
---

[2071. Maximum Number of Tasks You Can Assign](https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/description) hard
[blog post](https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/solutions/6704221/kotlin-rust-by-samoylenkodmitry-avey/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01052025-2071-maximum-number-of-tasks?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-nHm8N4a38M)
![1.webp](/assets/leetcode_daily_images/2e8d7d1b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/975

#### Problem TLDR

Tasks can be done by workers with pills #hard #binary_search

#### Intuition

Didn't solve myself.

Thought process:

```j

    // 5 9 8 5 9      5 5 8 9 9       p=1 s=5
    // 1 6 4 2 6      1 2 4 6 6

    // 5 5 8 9 9
    // *         1+5 p=0 not optimal
    //   *       6
    //           -
    //               maybe start with whats already fits?
    //               or just DP by (i,j,pills) - n^3
    //
    // idea: real pointer + heap of skipped enchanced values (optimal?)-not optimal
    // idea: all in heap, real + enchanced, use greedy (not optimal)

    // hint: first smallest k to the workers
    // use binary search
    // but how to assign k smallest to all workers?
    // 5 5 8 9 9 - tasks              1 2 4 6 6  - workers     p=1 s=5
    // . . k, all must fit - key idea
    //        still, how to optimally give the pills?

    // 5 5 8            1 2 3 4 5 6  p=1 s=5    can assign all tasks to workers?
    //                                          start with biggest
    //                      8                   assign it to smallest+pill

```

The hint: the is a greedy way to check if `x` workers can or can not do `x` tasks.

Now, after the hint I was able to discover the working greedy algorithm:
* start with the biggest task and worker
* if it works - it works
* if it not, *don't give the pill to that worker*, instead, find the weakest worker that will do that with pill

Another weak point of mine was: how to actually implement this search for the weakest worker? Do we have to track the used workers somehow? Is the solution becomes O(workers^2)

Thats where I gave up and looked for the answer:
* put all the workers in a queue by their `pill potential`
* if task is doable - take from the front of the queue (meaning the strongest worker)
* if its not - consume the pill and the weakest worker (back of the `pill potential queue`)

#### Approach

* some greedy ideas are not working, we have to try different examples
* the `implementation` can be the hardest part even if you know the algorithm
* 1 hr - brain can't give its full power after this line (for me)

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 180ms
    fun maxTaskAssign(t: IntArray, ws: IntArray, pills: Int, s: Int): Int {
        t.sort(); ws.sort(); var lo = 0; var hi = t.lastIndex; var res = -1
        while (lo <= hi) {
            val m = (lo + hi) / 2
            var j = ws.lastIndex; var p = pills; var good = true; val q = ArrayDeque<Int>()
            for (i in m downTo 0) {
                while (j >= 0 && j >= ws.lastIndex - m && ws[j] + s >= t[i]) q += ws[j--]
                if (q.size < 1) { good = false; break }
                if (q.first() >= t[i]) q.removeFirst() else if (--p < 0) { good = false; break } else q.removeLast()
            }
            if (good) { res = max(res, m); lo = m + 1 } else hi = m - 1
        }
        return res + 1
    }

```
```rust

// 15ms
    pub fn max_task_assign(mut t: Vec<i32>, mut ws: Vec<i32>, pills: i32, s: i32) -> i32 {
        t.sort_unstable(); ws.sort_unstable(); let (mut lo, mut hi, mut r) = (0, t.len() as i32 - 1, -1);
        while lo <= hi {
            let m = (lo + hi) as usize / 2;
            let (mut j, mut p, mut good, mut q) = (ws.len() - 1, pills, true, VecDeque::new());
            for i in (0..=m).rev() {
                while j < ws.len() && j >= ws.len() - m - 1 && ws[j] + s >= t[i] { q.push_back(ws[j]); j -= 1 }
                if q.len() < 1 { good = false; break }
                if *q.front().unwrap() >= t[i] { q.pop_front(); } else if p < 1 { good = false; break }
                else { q.pop_back(); p -= 1 }
            }
            if good { r = r.max(m as i32); lo = m as i32 + 1 } else { hi = m as i32 - 1 }
        }
        r + 1
    }

```
```c++

// 66ms
    int maxTaskAssign(vector<int>& t, vector<int>& ws, int pills, int s) {
        int l = 0, r = min(size(t), size(ws)); sort(begin(t), end(t)); sort(begin(ws), end(ws));
        while (l < r) {
            int m = (l + r + 1) / 2, p = pills, j = size(ws) - 1, g = 1; deque<int> q;
            for (int i = m - 1; i >= 0 && g; --i) {
                while (j >= 0 && j >= size(ws) - m && ws[j] + s >= t[i]) q.push_back(ws[j--]);
                if (!size(q)) g = 0; else
                if (q.front() >= t[i]) q.pop_front(); else if (--p < 0) g = 0; else q.pop_back();
            }
            if (g) l = m; else r = m - 1;
        } return l;
    }

```

