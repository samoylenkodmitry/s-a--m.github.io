---
layout: leetcode-entry
title: "979. Distribute Coins in Binary Tree"
permalink: "/leetcode/problem/2024-05-18-979-distribute-coins-in-binary-tree/"
leetcode_ui: true
entry_slug: "2024-05-18-979-distribute-coins-in-binary-tree"
---

[979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/distribute-coins-in-binary-tree/solutions/5173456/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18052024-979-distribute-coins-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-bec2qToKoM)
![2024-05-18_09-23.webp](/assets/leetcode_daily_images/3305f4dc.webp)
https://youtu.be/-bec2qToKoM
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/606

#### Problem TLDR

Min moves to spread the coins across the tree #medium #dfs #tree

#### Intuition

Let's observe some examples:
![2024-05-18_08-32.webp](/assets/leetcode_daily_images/982dc687.webp)
Some observations:
* each coin moves individually, even if we move `2` coins at once, it makes no difference to the total moves
* eventually, every node will have exactly `1` coin
We can use abstract `flow`:
* `0` coins at leaves have `flow = -1`, because they are attracting coin
* flow is accumulating from children to parent, so we can compute it independently for the `left` and `right` nodes
* total moves count is sign-independent sum of total flow: we count both negative and positive moves

#### Approach

* for Rust there is an interesting way to use `Option` in combinations with `?` operation that will return `None`; it helps to reduce the code size

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(log(n))$$, for the recursion depth

#### Code

```kotlin

    fun distributeCoins(root: TreeNode?): Int {
        var res = 0
        fun dfs(n: TreeNode?): Int = n?.run {
          (dfs(left) + dfs(right) + `val` - 1).also { res += abs(it) }} ?: 0
        dfs(root)
        return res
    }

```
```rust

    pub fn distribute_coins(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(n: &Option<Rc<RefCell<TreeNode>>>, res: &mut i32) -> Option<i32> {
            let n = n.as_ref()?; let n = n.borrow();
            let flow = dfs(&n.left, res).unwrap_or(0) + dfs(&n.right, res).unwrap_or(0) + n.val - 1;
            *res += flow.abs(); Some(flow)
        }
        let mut res = 0; dfs(&root, &mut res); res
    }

```

