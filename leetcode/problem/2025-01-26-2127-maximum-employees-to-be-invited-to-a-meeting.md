---
layout: leetcode-entry
title: "2127. Maximum Employees to Be Invited to a Meeting"
permalink: "/leetcode/problem/2025-01-26-2127-maximum-employees-to-be-invited-to-a-meeting/"
leetcode_ui: true
entry_slug: "2025-01-26-2127-maximum-employees-to-be-invited-to-a-meeting"
---

[2127. Maximum Employees to Be Invited to a Meeting](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/description/) hard
[blog post](https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting/solutions/6331395/kotlin-rust-by-samoylenkodmitry-s4cs/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26012025-2127-maximum-employees-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GiZialiwbuE)
![1.webp](/assets/leetcode_daily_images/11dddb2e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/876

#### Problem TLDR

Max connected siblings in graph #hard #dfs #toposort

#### Intuition

Failed to solve this.

This problem require to deduct several insights:
* individual cycles can live together
* there are two types: big cycles and small cycle with tail
* cycles are not intersect by definition (otherwise they merge)

Big cycles is when there are no small cycles in them.
Small cycle is sibling-cycle: a <-> b and all their followers.

#### Approach

* feel free to steal the code after 1.5 hours of trying
* toposort leftovers are cycles

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maximumInvitations(fav: IntArray): Int {
        val g = Array(fav.size) { ArrayList<Int>() }
        for (i in fav.indices) if (fav[fav[i]] != i) g[fav[i]] += i
        fun dfs(i: Int): Int = 1 + (g[i].maxOfOrNull { dfs(it) } ?: 0)
        val vis = IntArray(fav.size)
        return max(
            fav.indices.sumOf { if (fav[fav[it]] == it) dfs(it) else 0 },
            fav.indices.maxOf { i ->
                var cycle = 0; var j = i; var k = i
                while (vis[j] < 1) { cycle++; vis[j]++; j = fav[j] }
                while (j != k) { cycle--; k = fav[k] }
                cycle
            }
        )
    }

```
```rust

    pub fn maximum_invitations(fav: Vec<i32>) -> i32 {
        let mut deg = vec![0; fav.len()]; let mut path = deg.clone();
        for i in 0..fav.len() { deg[fav[i] as usize] += 1 }
        let mut q = VecDeque::from_iter((0..fav.len()).filter(|&i| deg[i] == 0));
        while let Some(i) = q.pop_front() {
            let j = fav[i] as usize; path[j] = path[i] + 1;
            deg[j] -= 1; if deg[j] == 0 { q.push_back(j) }
        }
        let (mut path_sum, mut cycle_max) = (0, 0);
        for i in 0..fav.len() {
            let (mut cycle, mut j) = (0, i);
            while deg[j] > 0 { deg[j] = 0; j = fav[j] as usize; cycle += 1 }
            if cycle == 2 {
                path_sum += 2 + path[i] + path[fav[i] as usize]
            } else {
                cycle_max = cycle_max.max(cycle)
            }
        }
        path_sum.max(cycle_max)
    }

```
```c++

    int maximumInvitations(vector<int>& f) {
        int n = size(f), s = 0, m = 0; vector<vector<int>>g(n); vector<int> v(n);
        for (int i = 0; i < n; ++i) if (f[f[i]] != i) g[f[i]].push_back(i);
        auto d = [&](this auto const& d, int i) -> int {
            int x = 0; for (int j: g[i]) x = max(x, d(j)); return 1 + x;};
        for (int i = 0; i < n; ++i) {
            if (f[f[i]] == i) { s += d(i); continue; }
            int c = 0, j = i, k = i;
            while (!v[j]) ++c, ++v[j], j = f[j];
            while (j != k) --c, k = f[k];
            m = max(m, c);
        } return max(s, m);
    }

```

