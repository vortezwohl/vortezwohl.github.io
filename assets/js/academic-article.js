document.addEventListener("DOMContentLoaded", () => {
  const root = document.querySelector("[data-academic-article]");
  if (!root) {
    return;
  }

  const detailMap = new Map();
  document.querySelectorAll("[data-pattern-detail]").forEach((panel) => {
    detailMap.set(panel.dataset.patternDetail, panel);
  });

  const activatePattern = (patternId) => {
    document.querySelectorAll("[data-pattern-tab]").forEach((button) => {
      const isActive = button.dataset.patternTab === patternId;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-selected", String(isActive));
    });

    detailMap.forEach((panel, key) => {
      const isActive = key === patternId;
      panel.classList.toggle("is-active", isActive);
      panel.hidden = !isActive;
    });
  };

  const firstTab = document.querySelector("[data-pattern-tab]");
  if (firstTab) {
    activatePattern(firstTab.dataset.patternTab);
  }

  document.querySelectorAll("[data-pattern-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      const patternId = button.dataset.patternTab;
      activatePattern(patternId);
      const anchor = document.getElementById(`pattern-${patternId}`);
      if (anchor) {
        anchor.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  const demoRunners = {
    strategy(button) {
      const amount = Number(button.dataset.amount || 1000);
      const rate = Number(button.dataset.rate || 0.15);
      return `原价 ${amount} 元\n策略切换为 ${Math.round(rate * 100)}% 折扣\n最终价格 = ${amount * (1 - rate)} 元`;
    },
    observer() {
      return [
        "EventBus.publish('order_paid')",
        "-> inventory.reserve()",
        "-> coupon.issue()",
        "-> email.notify()",
        "特点：发布方不依赖任何一个具体订阅者",
      ].join("\n");
    },
    factory() {
      return [
        "creator = HtmlReportCreator()",
        "report = creator.create_report()",
        "report.render() -> <p>ok</p>",
        "变化点被隔离在 creator 里",
      ].join("\n");
    },
    abstract_factory() {
      return [
        "theme = DarkThemeFactory()",
        "theme.create_button() -> dark-button",
        "theme.create_dialog() -> dark-dialog",
        "整套组件保持风格一致",
      ].join("\n");
    },
    singleton() {
      return [
        "a = AppConfig()",
        "b = AppConfig()",
        "a is b -> True",
        "提示：生产中更常让 DI 容器管理 singleton scope",
      ].join("\n");
    },
    adapter() {
      return [
        "legacy.send_sms(phone, text)",
        "SmsAdapter.send(user, msg)",
        "旧接口不改，新系统只认识 Notifier.send()",
      ].join("\n");
    },
    decorator() {
      return [
        "BaseSender('hello')",
        "-> TimestampDecorator",
        "-> EncryptDecorator",
        "输出: send:[2026-05-22] olleh",
      ].join("\n");
    },
    proxy() {
      return [
        "AccessProxy.read('report.csv') -> allowed",
        "AccessProxy.read('salary.xlsx') -> PermissionError",
        "同一接口，额外加入访问控制",
      ].join("\n");
    },
    template_method() {
      return [
        "run()",
        "-> extract()",
        "-> transform()",
        "-> load()",
        "骨架稳定，局部步骤可替换",
      ].join("\n");
    },
    facade() {
      return [
        "OrderFacade.place('alice', 'book-1', 99.0)",
        "-> reserve inventory",
        "-> charge payment",
        "-> create shipping",
        "调用方只面对一个统一入口",
      ].join("\n");
    },
  };

  document.querySelectorAll("[data-demo-run]").forEach((button) => {
    button.addEventListener("click", () => {
      const demoName = button.dataset.demoRun;
      const panel = button.closest("[data-demo-panel]");
      const output = panel ? panel.querySelector("[data-demo-output]") : null;
      if (!output) {
        return;
      }

      const renderer = demoRunners[demoName];
      output.textContent = typeof renderer === "function" ? renderer(button) : "No demo renderer.";
    });
  });

  document.querySelectorAll("[data-copy-demo]").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.dataset.copyDemo;
      const source = document.getElementById(targetId);
      if (!source) {
        return;
      }

      try {
        await navigator.clipboard.writeText(source.textContent || "");
        const original = button.textContent;
        button.textContent = "已复制";
        window.setTimeout(() => {
          button.textContent = original;
        }, 1200);
      } catch (error) {
        button.textContent = "复制失败";
      }
    });
  });
});
